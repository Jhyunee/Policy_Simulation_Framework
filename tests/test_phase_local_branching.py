from __future__ import annotations

import unittest
from unittest.mock import patch

from models.schemas import (
    BranchMechanismAnchor,
    ConcretePersona,
    PhaseInnerSpec,
    PhaseRoundResult,
    PhaseVariableSpec,
    PolicyContext,
    ToCInnerSchema,
)
from models.prompts import (
    PHASE_LOCAL_BRANCH_DIRECTIVES_EN,
    PHASE_LOCAL_BRANCH_DIRECTIVES_KO,
)
from stage2_toc.branching import (
    generate_phase_children,
    mechanism_coverage_level,
    mechanism_planner_mode,
)
from stage2_toc.modules import (
    _format_branch_mechanism_anchor,
    _phase_local_transition_directive,
    _phase_round,
    resolve_configured_posting_order,
)
from scripts.generate_phase_branch_tree import (
    _is_baseline_spine_node,
    generation_plan,
    pending_parent_ids,
)


def policy_context() -> PolicyContext:
    return PolicyContext(
        policy_name="Test policy",
        policy_domain="research",
        policy_type="grant",
        delivery_mechanism="award",
        eligibility_logic="eligible applicants",
        resource_logic="fixed budget",
        theory_of_change_summary="funding enables activity and outcomes",
        simulation_relevance="test",
    )


def schema() -> ToCInnerSchema:
    return ToCInnerSchema(phases=[PhaseInnerSpec(
        phase="Inputs",
        prediction_fields=[PhaseVariableSpec(
            name="funded_slots", unit="count", type="int", description="Funded slots",
        )],
        discussion_points=["How many slots are usable?"],
    )])


class PhaseLocalBranchingTest(unittest.TestCase):
    def setUp(self):
        self.personas = [ConcretePersona(stakeholder_type="Applicant", name="A")]
        self.config = {
            "stage2_toc": {
                "prompt_variant": "ko_natural",
                "phase_interaction_mode": "independent",
                "stance": "conservative",
            }
        }

    def test_phase_local_prompt_ignores_legacy_global_stance(self):
        calls = []

        def fake_completion(messages, config):
            calls.append(messages)
            if "참여자 발언 전체" in messages[1]["content"]:
                return '{"summary":"부모에서 이어진 국소 전이를 요약한다."}'
            return '{"prediction_values":{"funded_slots":10},"narrative":"조건이 작동한다.","evidence":["근거 1","근거 2","근거 3"]}'

        with patch("stage2_toc.modules.chat_completion", side_effect=fake_completion):
            result = _phase_round(
                direction="forward",
                phase="Inputs",
                phase_spec=schema().phases[0],
                personas=self.personas,
                policy_context=policy_context(),
                simulation_target="funded slots",
                prior_phases=[],
                policy_goal_seed=None,
                config=self.config,
                inner_schema=schema(),
                transition_mode="enabling",
                fixed_posting_order=["A"],
            )

        self.assertEqual(result.posting_order, ["A"])
        self.assertIn("LOCAL TRANSITION — ENABLING", calls[0][1]["content"])
        self.assertIn("BRANCH INHERITANCE CONTRACT", calls[0][1]["content"])
        self.assertNotIn("ADDITIONAL FRAMING", calls[0][0]["content"])

    def test_english_phase_local_prompt_preserves_same_branch_contract(self):
        calls = []
        english_config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "phase_interaction_mode": "independent",
                "stance": "optimistic",
            }
        }

        def fake_completion(messages, config):
            calls.append(messages)
            if "All participant posts" in messages[1]["content"]:
                return '{"summary":"A bottleneck weakens the inherited causal connection."}'
            return '{"prediction_values":{"funded_slots":4},"narrative":"A bottleneck constrains delivery.","evidence":["evidence 1","evidence 2","evidence 3"]}'

        with patch("stage2_toc.modules.chat_completion", side_effect=fake_completion):
            result = _phase_round(
                direction="forward",
                phase="Inputs",
                phase_spec=schema().phases[0],
                personas=self.personas,
                policy_context=policy_context(),
                simulation_target="funded slots",
                prior_phases=[],
                policy_goal_seed=None,
                config=english_config,
                inner_schema=schema(),
                transition_mode="constraining",
                fixed_posting_order=["A"],
            )

        self.assertEqual(result.phase_summary, "A bottleneck weakens the inherited causal connection.")
        self.assertIn("LOCAL TRANSITION — CONSTRAINING", calls[0][1]["content"])
        self.assertIn("BRANCH INHERITANCE CONTRACT", calls[0][1]["content"])
        self.assertIn("reversal away from the policy objective", calls[0][1]["content"])
        self.assertIn("clear, natural English", calls[0][0]["content"])
        self.assertNotIn("ADDITIONAL FRAMING", calls[0][0]["content"])

    def test_intensity_experiment_changes_only_enabling_and_constraining(self):
        config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "branch_prompt_experiment": "intensity",
            }
        }
        enabling = _phase_local_transition_directive(config, "enabling")
        baseline = _phase_local_transition_directive(config, "baseline")
        constraining = _phase_local_transition_directive(config, "constraining")

        self.assertIn("material, rather than merely marginal", enabling)
        self.assertIn("materially, rather than merely marginally", constraining)
        self.assertNotIn("BRANCH INTENSITY EXPERIMENT", baseline)

    def test_persona_constraint_experiment_applies_the_same_role_contract_to_all_siblings(self):
        config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "branch_prompt_experiment": "persona_constraint",
            }
        }
        directives = [
            _phase_local_transition_directive(config, mode)
            for mode in ("enabling", "baseline", "constraining")
        ]

        for directive in directives:
            self.assertIn("most binding role-specific constraint", directive)
            self.assertIn("generic system-wide bottleneck as the sole rationale", directive)

    def test_persona_mechanism_coverage_supports_diverse_grounded_pathways(self):
        config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "branch_prompt_experiment": "persona_mechanism_coverage",
            }
        }
        directives = [
            _phase_local_transition_directive(config, mode)
            for mode in ("enabling", "baseline", "constraining")
        ]

        for directive in directives:
            self.assertIn("different configuration of conditions", directive)
            self.assertIn("does not need to match", directive)
            self.assertIn("Do not list unrelated factors", directive)
            self.assertIn("fixed facts", directive)
            self.assertIn("conditional mechanisms", directive)
            self.assertIn("before deriving prediction_values", directive)
            self.assertIn("Do not sacrifice plausibility for novelty", directive)

    def test_persona_mechanism_coverage_has_matching_korean_contract(self):
        config = {
            "stage2_toc": {
                "prompt_variant": "ko_natural",
                "branch_prompt_experiment": "persona_mechanism_coverage",
            }
        }
        directive = _phase_local_transition_directive(config, "constraining")

        self.assertIn("서로 다른 조건 조합", directive)
        self.assertIn("다른 branch와 동일한 요인을 사용할 필요가 없습니다", directive)
        self.assertIn("고정 사실", directive)
        self.assertIn("조건부 메커니즘", directive)
        self.assertIn("먼저 메커니즘과 전파 과정을 정한 뒤", directive)

    def test_role_constraint_activation_varies_one_fixed_constraint_by_branch(self):
        config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "branch_prompt_experiment": "role_constraint_activation",
            }
        }
        enabling = _phase_local_transition_directive(config, "enabling")
        baseline = _phase_local_transition_directive(config, "baseline")
        constraining = _phase_local_transition_directive(config, "constraining")

        for directive in (enabling, baseline, constraining):
            self.assertIn("exactly one primary binding constraint", directive)
            self.assertIn("Do not invent a new constraint", directive)
            self.assertIn("Do not revise parent facts or quantities", directive)
        self.assertIn("relaxes, offsets, or makes it more manageable", enabling)
        self.assertIn("persists at the inherited base-rate level", baseline)
        self.assertIn("intensifies it, reduces feasible action", constraining)

    def test_siblings_share_parent_hash_and_posting_order(self):
        def fake_round(**kwargs):
            return PhaseRoundResult(
                phase=kwargs["phase"],
                direction="forward",
                initial_posts=[],
                revised_posts=[],
                phase_summary=kwargs["transition_mode"],
                posting_order=kwargs["fixed_posting_order"],
            )

        with patch("stage2_toc.branching._phase_round", side_effect=fake_round):
            children = generate_phase_children(
                parent_id="root",
                prior_phases=[],
                personas=self.personas,
                inner_schema=schema(),
                policy_context=policy_context(),
                simulation_target="funded slots",
                config=self.config,
            )

        self.assertEqual([c.transition_mode for c in children], [
            "enabling", "baseline", "constraining",
        ])
        self.assertEqual(len({c.parent_context_hash for c in children}), 1)
        self.assertTrue(all(c.phase_result.posting_order == ["A"] for c in children))

    def test_optional_planner_adds_shared_anchors_only_to_ec_nodes(self):
        planner_config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "phase_interaction_mode": "independent",
                "branch_mechanism_planner": {
                    "enabled": True,
                    "coverage_level_override": "material",
                },
            }
        }

        def fake_completion(messages, config):
            return """{
              "mechanism_family": "capacity",
              "trigger": "shared trigger",
              "affected_causal_link": "input -> output",
              "expected_propagation": ["first", "second"],
              "parent_support": "supported by parent",
              "plausibility_limit": "does not assume policy repeal"
            }"""

        def fake_round(**kwargs):
            return PhaseRoundResult(
                phase=kwargs["phase"],
                direction="forward",
                initial_posts=[],
                revised_posts=[],
                phase_summary=kwargs["transition_mode"],
                posting_order=kwargs["fixed_posting_order"],
            )

        with (
            patch("stage2_toc.branching.chat_completion", side_effect=fake_completion) as planner,
            patch("stage2_toc.branching._phase_round", side_effect=fake_round) as phase_round,
        ):
            children = generate_phase_children(
                parent_id="root",
                prior_phases=[],
                personas=self.personas,
                inner_schema=schema(),
                policy_context=policy_context(),
                simulation_target="funded slots",
                config=planner_config,
            )

        self.assertEqual(planner.call_count, 2)
        anchors = {child.transition_mode: child.mechanism_anchor for child in children}
        self.assertIsNotNone(anchors["enabling"])
        self.assertIsNone(anchors["baseline"])
        self.assertIsNotNone(anchors["constraining"])
        self.assertEqual(anchors["enabling"].coverage_level, "material")
        passed = {
            call.kwargs["transition_mode"]: call.kwargs["mechanism_anchor"]
            for call in phase_round.call_args_list
        }
        self.assertIs(passed["enabling"], anchors["enabling"])
        self.assertIsNone(passed["baseline"])

    def test_joint_sibling_planner_plans_ebc_once_and_injects_branch_deltas(self):
        planner_config = {
            "stage2_toc": {
                "prompt_variant": "en_natural",
                "phase_interaction_mode": "independent",
                "branch_mechanism_planner": {
                    "enabled": True,
                    "mode": "joint_siblings",
                    "coverage_level_override": "material",
                },
            }
        }

        def fake_completion(messages, config):
            self.assertIn("Repeating an inherited constraint is not a branch distinction", messages[1]["content"])
            return """{
              "common_parent_conditions": ["The inherited budget remains fixed."],
              "sibling_contrast": "Usable capacity expands, persists, or becomes blocked.",
              "branches": {
                "enabling": {
                  "mechanism_family": "capacity release",
                  "trigger": "Dormant capacity becomes usable.",
                  "affected_causal_link": "budget -> usable slots",
                  "expected_propagation": ["More slots open."],
                  "parent_support": "The parent has funded but unused capacity.",
                  "plausibility_limit": "The budget does not increase.",
                  "expected_variable_effects": [{
                    "variable": "funded_slots", "direction": "increase",
                    "timing": "current_phase", "strength": "material",
                    "rationale": "Existing capacity becomes usable."
                  }]
                },
                "baseline": {
                  "mechanism_family": "inherited base-rate continuation",
                  "trigger": "No new shock; current utilization persists.",
                  "affected_causal_link": "budget -> expected slots",
                  "expected_propagation": ["Slot delivery remains near the inherited rate."],
                  "parent_support": "No changed condition is observed.",
                  "plausibility_limit": "Ordinary uncertainty remains.",
                  "expected_variable_effects": [{
                    "variable": "funded_slots", "direction": "stable",
                    "timing": "current_phase", "strength": "reference",
                    "rationale": "The inherited base rate continues."
                  }]
                },
                "constraining": {
                  "mechanism_family": "capacity blockage",
                  "trigger": "Administrative processing blocks funded capacity.",
                  "affected_causal_link": "budget -> delivered slots",
                  "expected_propagation": ["Fewer slots become usable."],
                  "parent_support": "The parent identifies processing risk.",
                  "plausibility_limit": "The program is not repealed.",
                  "expected_variable_effects": [{
                    "variable": "funded_slots", "direction": "decrease",
                    "timing": "current_phase", "strength": "material",
                    "rationale": "The bottleneck blocks delivery."
                  }]
                }
              }
            }"""

        def fake_round(**kwargs):
            return PhaseRoundResult(
                phase=kwargs["phase"],
                direction="forward",
                initial_posts=[],
                revised_posts=[],
                phase_summary=kwargs["transition_mode"],
                posting_order=kwargs["fixed_posting_order"],
            )

        with (
            patch("stage2_toc.branching.chat_completion", side_effect=fake_completion) as planner,
            patch("stage2_toc.branching._phase_round", side_effect=fake_round) as phase_round,
        ):
            children = generate_phase_children(
                parent_id="root",
                prior_phases=[],
                personas=self.personas,
                inner_schema=schema(),
                policy_context=policy_context(),
                simulation_target="funded slots",
                config=planner_config,
            )

        self.assertEqual(planner.call_count, 1)
        anchors = {child.transition_mode: child.mechanism_anchor for child in children}
        self.assertEqual(set(anchors), {"enabling", "baseline", "constraining"})
        self.assertTrue(all(anchor is not None for anchor in anchors.values()))
        self.assertEqual(anchors["baseline"].coverage_level, "reference")
        self.assertEqual(anchors["enabling"].expected_variable_effects[0].direction, "increase")
        self.assertEqual(anchors["constraining"].expected_variable_effects[0].direction, "decrease")
        self.assertEqual(
            anchors["enabling"].common_parent_conditions,
            anchors["baseline"].common_parent_conditions,
        )
        passed = {
            call.kwargs["transition_mode"]: call.kwargs["mechanism_anchor"]
            for call in phase_round.call_args_list
        }
        self.assertIs(passed["baseline"], anchors["baseline"])

    def test_planner_mode_defaults_to_legacy_per_transition(self):
        self.assertEqual(mechanism_planner_mode({}), "per_transition")
        self.assertEqual(
            mechanism_planner_mode({
                "stage2_toc": {"branch_mechanism_planner": {"mode": "joint_siblings"}}
            }),
            "joint_siblings",
        )

    def test_mechanism_coverage_level_is_deterministic_and_overrideable(self):
        config = {"stage2_toc": {"branch_mechanism_planner": {}}}
        first = mechanism_coverage_level(
            parent_id="root",
            parent_hash="abc",
            phase_index=1,
            transition_mode="enabling",
            config=config,
        )
        second = mechanism_coverage_level(
            parent_id="root",
            parent_hash="abc",
            phase_index=1,
            transition_mode="enabling",
            config=config,
        )
        self.assertEqual(first, second)

        config["stage2_toc"]["branch_mechanism_planner"]["coverage_level_override"] = "boundary"
        self.assertEqual(
            mechanism_coverage_level(
                parent_id="root",
                parent_hash="abc",
                phase_index=1,
                transition_mode="enabling",
                config=config,
            ),
            "boundary",
        )

    def test_mechanism_anchor_preserves_field_units_and_scale(self):
        anchor = BranchMechanismAnchor(
            transition_mode="enabling",
            coverage_level="boundary",
            mechanism_family="capacity",
            trigger="Operational capacity becomes available.",
            affected_causal_link="capacity -> completed cases",
            expected_propagation=["More cases are processed."],
            parent_support="The parent identifies unused capacity.",
            plausibility_limit="Capacity remains finite.",
        )

        rendered = _format_branch_mechanism_anchor(anchor)

        self.assertIn("not a numerical magnitude", rendered)
        self.assertIn("declared unit and plausible scale", rendered)
        self.assertIn("without a denominator grounded", rendered)

    def test_configured_posting_order_resolves_role_ids(self):
        personas = [
            ConcretePersona(
                persona_id="p-provider",
                name="Provider A",
                stakeholder_type="provider",
                role_prototype_id="provider",
            ),
            ConcretePersona(
                persona_id="p-patient",
                name="Patient A",
                stakeholder_type="patient",
                role_prototype_id="patient",
            ),
        ]
        config = {
            "stage2_toc": {
                "fixed_posting_order": ["patient", "provider"],
            }
        }

        self.assertEqual(
            resolve_configured_posting_order(personas, config),
            ["Patient A", "Provider A"],
        )

    def test_configured_posting_order_requires_complete_coverage(self):
        personas = [
            ConcretePersona(name="A", stakeholder_type="patient"),
            ConcretePersona(name="B", stakeholder_type="provider"),
        ]
        config = {"stage2_toc": {"fixed_posting_order": ["patient"]}}

        with self.assertRaisesRegex(ValueError, "include every participant"):
            resolve_configured_posting_order(personas, config)

    def test_baseline_transition_is_parent_conditioned_without_extra_assumptions(self):
        english = PHASE_LOCAL_BRANCH_DIRECTIVES_EN["baseline"]
        korean = PHASE_LOCAL_BRANCH_DIRECTIVES_KO["baseline"]

        self.assertIn("most plausible pathway from the parent's current state", english)
        self.assertIn("new shock or change in conditions", english)
        self.assertNotIn("attrition", english)
        self.assertNotIn("midpoint", english)
        self.assertNotIn("weaken", english)

        self.assertIn("parent의 현재 상태에서 가장 개연성 높게", korean)
        self.assertIn("새로운 충격이나 조건 변화", korean)
        self.assertNotIn("이탈", korean)
        self.assertNotIn("중간값", korean)
        self.assertNotIn("약화", korean)

    def test_generation_plan_matches_complete_ternary_tree(self):
        fixed_inputs = generation_plan(phase_count=5, persona_count=5, branch_inputs=False)
        self.assertEqual(fixed_inputs, {
            "generated_nodes": 120,
            "complete_paths": 81,
            "estimated_llm_calls": 720,
        })
        branched_inputs = generation_plan(phase_count=5, persona_count=5, branch_inputs=True)
        self.assertEqual(branched_inputs["generated_nodes"], 363)
        self.assertEqual(branched_inputs["complete_paths"], 243)
        self.assertEqual(branched_inputs["estimated_llm_calls"], 2178)

    def test_generation_plan_supports_single_transition_path(self):
        baseline_only = generation_plan(
            phase_count=5,
            persona_count=5,
            branch_inputs=False,
            branch_factor=1,
        )
        self.assertEqual(baseline_only, {
            "generated_nodes": 4,
            "complete_paths": 1,
            "estimated_llm_calls": 24,
        })

    def test_generation_plan_supports_baseline_spine_with_sibling_contrasts(self):
        baseline_spine = generation_plan(
            phase_count=5,
            persona_count=5,
            branch_inputs=False,
            expansion_strategy="baseline_spine",
        )
        self.assertEqual(baseline_spine, {
            "generated_nodes": 12,
            "complete_paths": 3,
            "estimated_llm_calls": 72,
        })

    def test_baseline_spine_pending_queue_excludes_off_spine_siblings(self):
        tree = {
            "completed_parent_ids": ["root"],
            "nodes": {
                "root": {"node_id": "root", "phase_index": 0},
                "root/enabling": {"node_id": "root/enabling", "phase_index": 1},
                "root/baseline": {"node_id": "root/baseline", "phase_index": 1},
                "root/constraining": {"node_id": "root/constraining", "phase_index": 1},
                "root/baseline/baseline": {
                    "node_id": "root/baseline/baseline",
                    "phase_index": 2,
                },
            },
        }

        self.assertTrue(_is_baseline_spine_node("root/baseline/baseline"))
        self.assertFalse(_is_baseline_spine_node("root/baseline/enabling"))
        self.assertEqual(
            list(pending_parent_ids(tree, phase_count=5, expansion_strategy="baseline_spine")),
            ["root/baseline", "root/baseline/baseline"],
        )

    def test_generation_plan_counts_independent_revision_calls(self):
        revised = generation_plan(
            phase_count=5,
            persona_count=5,
            branch_inputs=False,
            branch_factor=1,
            interaction_mode="independent_revision",
        )
        self.assertEqual(revised["estimated_llm_calls"], 44)

    def test_generation_plan_counts_optional_planner_calls(self):
        planned = generation_plan(
            phase_count=5,
            persona_count=5,
            branch_inputs=False,
            planner_calls_per_parent=2,
        )
        self.assertEqual(planned["estimated_llm_calls"], 800)

        joint_planned = generation_plan(
            phase_count=5,
            persona_count=5,
            branch_inputs=False,
            planner_calls_per_parent=1,
        )
        self.assertEqual(joint_planned["estimated_llm_calls"], 760)


if __name__ == "__main__":
    unittest.main()
