from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.schemas import PersonaProfile, PolicyContext, StakeholderGroup
from persona.name_utils import infer_persona_name
from persona.simulation_schema import (
    POLICY_ROLE_CONTEXT_FIELDS,
    SIMULATION_PERSONA_FIELDS,
)
from stage1.mechanism_panel import (
    attach_role_prototype,
    build_role_prototypes,
    select_mechanism_panel,
)
from stage1.modules import _expand_with_kg_gen
from stage1.pipeline import _project_persona_for_stage2
from stage1.pipeline import _refine_selected_personas
from stage1.pipeline import run_stage1
from stage2_toc.pipeline import _build_personas_from_stage1_personas
from scripts.refine_scenario_personas import _restore_pre_refine_narratives


def config() -> dict:
    return {
        "llm": {"temperature": 0.3},
        "persona": {
            "selection_mode": "mechanism_panel",
            "simulation_sample_seed": 42,
            "mechanism_panel": {
                "candidates_per_role": 2,
                "strict_role_coverage": True,
            },
            "kg_gen": {"country": "usa", "personas_per_entity": 2},
            "fallback_to_llm": False,
        },
    }


def policy_context() -> PolicyContext:
    return PolicyContext(
        policy_name="No Surprises Act",
        policy_domain="health policy",
        policy_type="regulation",
        delivery_mechanism="billing protections",
        eligibility_logic="covered patients",
        resource_logic="administrative implementation",
        theory_of_change_summary="rules alter billing and dispute resolution",
        simulation_relevance="surprise billing",
    )


class FakeEntity:
    uuid = "RegulatoryAgency:HHS"
    name = "HHS Regulatory Agency"
    summary = "Federal agency implementing policy"
    attributes = {"iad_category": "Participants", "agent_modality": "organization"}


class MechanismPanelTest(unittest.TestCase):
    def _graph_path(self, directory: str) -> Path:
        nodes = [
            {"id": "BeneficiaryOrEnrollee:Patient", "name": "Patient", "type": "BeneficiaryOrEnrollee", "summary": "Protected patient", "is_actor": False, "persona_eligible": False},
            {"id": "NonparticipatingProvider:Provider", "name": "Nonparticipating Provider", "type": "NonparticipatingProvider", "summary": "Provider subject to billing rules", "is_actor": False, "persona_eligible": False},
            {"id": "GroupHealthPlanOrIssuer:Issuer", "name": "Health Plan or Issuer", "type": "GroupHealthPlanOrIssuer", "summary": "Claims payer", "is_actor": True, "persona_eligible": True},
            {"id": "RegulatoryAgency:HHS", "name": "HHS Regulatory Agency", "type": "RegulatoryAgency", "summary": "Federal regulator", "is_actor": True, "persona_eligible": True},
            {"id": "IndependentDisputeResolutionEntity:IDR", "name": "IDR Entity", "type": "IndependentDisputeResolutionEntity", "summary": "Independent dispute resolver", "is_actor": False, "persona_eligible": False},
            {"id": "PolicyAction:PatientNotice", "name": "Patient notice action", "type": "PolicyAction", "summary": "A patient receives a notice", "is_actor": False, "persona_eligible": False},
        ]
        edges = [
            {"id": f"e{i}", "source": node["id"], "target": "PolicyAction:PatientNotice", "type": "ACTS_ON", "fact": f"{node['name']} performs a grounded policy action"}
            for i, node in enumerate(nodes[:5], 1)
        ]
        path = Path(directory) / "policy_graph.json"
        path.write_text(json.dumps({"nodes": nodes, "edges": edges}), encoding="utf-8")
        return path

    def test_role_builder_recovers_affected_actors_without_reclassifying_policy_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            prototypes, overlay = build_role_prototypes(self._graph_path(tmp), config())

        self.assertEqual([p.prototype_id for p in prototypes], [
            "patient", "provider", "payer", "regulator", "mediator",
        ])
        overlay_by_id = {item["entity_id"]: item for item in overlay}
        self.assertTrue(overlay_by_id["BeneficiaryOrEnrollee:Patient"]["panel_role_candidate"])
        self.assertFalse(overlay_by_id["BeneficiaryOrEnrollee:Patient"]["source_persona_eligible"])
        self.assertNotIn("PolicyAction:PatientNotice", overlay_by_id)

    def test_panel_selection_and_stage2_projection_keep_only_compact_role_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            prototypes, _ = build_role_prototypes(self._graph_path(tmp), config())

        groups = []
        for prototype in prototypes:
            candidates = []
            for index in range(2):
                persona = PersonaProfile(
                    representative_type=f"{prototype.label} candidate {index}",
                    core_objectives=["full objective that remains in the audit artifact"],
                    core_constraints=["full constraint that remains in the audit artifact"],
                    kg_context={"large": ["not injected"] * 100},
                    demographics={
                        "professional_persona": f"Person {index} has relevant experience.",
                        "name": f"Person {index}",
                        "skills_and_expertise": "Relevant skill.",
                        "career_goals_and_ambitions": "Relevant goal.",
                        "occupation": f"occupation_{prototype.prototype_id}_{index}",
                        "province": f"state_{index}",
                        "match_score": 5 - index,
                        "used_fallback": False,
                    },
                )
                attach_role_prototype(persona, prototype)
                candidates.append(persona)
            groups.append(StakeholderGroup(stakeholder_type=prototype.domain_role, personas=candidates))

        selected, manifest = select_mechanism_panel(groups, prototypes, config())
        self.assertEqual(manifest["panel_size"], 5)
        self.assertTrue(manifest["ebc_blind_selection"])
        first_group = selected[0]
        record = _project_persona_for_stage2(first_group.personas[0], first_group.stakeholder_type)
        self.assertIn("policy_role_context", record)
        self.assertEqual(record["name"], "Person 0")
        self.assertLessEqual(set(record), set(SIMULATION_PERSONA_FIELDS))
        self.assertLessEqual(
            set(record["policy_role_context"]),
            set(POLICY_ROLE_CONTEXT_FIELDS),
        )
        self.assertLessEqual(len(record["policy_role_context"]["objectives"]), 2)
        self.assertNotIn("representative_type", record)
        self.assertNotIn("domain_role", record["policy_role_context"])
        self.assertNotIn("role_family", record["policy_role_context"])
        self.assertNotIn("kg_context", record)
        self.assertNotIn("compatibility_assessment", record)

        sim_data = {"scenarios": [{"participants": [record]}]}
        concrete = _build_personas_from_stage1_personas(sim_data)[0]
        self.assertEqual(concrete.policy_role_context, record["policy_role_context"])

    def test_panel_selection_honors_audited_candidate_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            prototypes, _ = build_role_prototypes(self._graph_path(tmp), config())

        groups = []
        for prototype in prototypes:
            candidates = []
            for index in range(2):
                persona = PersonaProfile(
                    representative_type=f"{prototype.label} candidate {index}",
                    core_objectives=["objective"],
                    core_constraints=["constraint"],
                    demographics={
                        "professional_persona": "Grounded profile.",
                        "skills_and_expertise": "Relevant skill.",
                        "career_goals_and_ambitions": "Relevant goal.",
                        "match_score": 10 - index,
                    },
                )
                attach_role_prototype(persona, prototype)
                candidates.append(persona)
            groups.append(StakeholderGroup(stakeholder_type=prototype.domain_role, personas=candidates))

        overridden = config()
        overridden["persona"]["mechanism_panel"]["candidate_overrides"] = {
            "patient": "Patient or covered beneficiary candidate 1"
        }
        overridden["persona"]["mechanism_panel"]["name_overrides"] = {
            "patient": "Audited Name"
        }
        selected, manifest = select_mechanism_panel(groups, prototypes, overridden)

        self.assertEqual(
            selected[0].personas[0].representative_type,
            "Patient or covered beneficiary candidate 1",
        )
        self.assertEqual(manifest["selections"][0]["selection_method"], "candidate_override")
        self.assertEqual(selected[0].personas[0].demographics["name"], "Audited Name")

    def test_organization_anchor_generates_human_representatives_in_new_mode(self):
        base = PersonaProfile(
            representative_type="HHS",
            core_objectives=["Implement the policy"],
            core_constraints=["Coordinate enforcement"],
        )
        response = json.dumps({"personas": [
            {
                "name": "Alex Morgan",
                "professional_persona": "Alex is a federal policy implementation officer.",
                "skills_and_expertise": "Regulatory implementation.",
                "career_goals_and_ambitions": "Improve policy delivery.",
                "sex": "Nonbinary",
                "age": 42,
                "occupation": "policy_officer",
                "province": "DC",
            },
            {
                "name": "Taylor Reed",
                "professional_persona": "Taylor manages federal health policy compliance.",
                "skills_and_expertise": "Compliance oversight.",
                "career_goals_and_ambitions": "Strengthen enforcement.",
                "sex": "Female",
                "age": 46,
                "occupation": "compliance_manager",
                "province": "MD",
            },
        ]})
        with patch("stage1.modules.chat_completion", return_value=response):
            personas = _expand_with_kg_gen(base, FakeEntity(), "RegulatoryAgency", policy_context(), config())

        self.assertEqual(len(personas), 2)
        self.assertTrue(all(
            persona.demographics["agent_modality"] == "organization_representative"
            for persona in personas
        ))

    def test_stage1_writes_fixed_panel_artifacts_and_one_runtime_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_path = self._graph_path(tmp)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            def fake_generate(*args, role_prototypes=None, **kwargs):
                groups = []
                for prototype in role_prototypes or []:
                    persona = PersonaProfile(
                        representative_type=f"Selected {prototype.label}",
                        core_objectives=["objective"],
                        core_constraints=["constraint"],
                        demographics={
                            "professional_persona": f"A grounded {prototype.domain_role} representative.",
                            "skills_and_expertise": "Policy-relevant expertise.",
                            "career_goals_and_ambitions": "Improve implementation.",
                            "candidate_index": 1,
                        },
                    )
                    attach_role_prototype(persona, prototype)
                    groups.append(StakeholderGroup(
                        stakeholder_type=prototype.domain_role,
                        personas=[persona],
                    ))
                return groups

            with (
                patch("stage1.pipeline.build_policy_context", return_value=policy_context()),
                patch("stage1.pipeline.generate_personas_from_policy_graph", side_effect=fake_generate),
                patch("stage1.pipeline._attach_policy_role_actions"),
                patch("stage1.pipeline.synthesize_simulation_participants", side_effect=lambda groups, *_: groups),
                patch("stage1.pipeline._refine_selected_personas") as refiner_mock,
            ):
                result = run_stage1(
                    config(),
                    run_dir,
                    "surprise billing reduction",
                    policy_graph_path=graph_path,
                )
            refiner_mock.assert_not_called()

            bundle = json.loads((run_dir / "simulation_personas.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema_version"], "simulation_persona_v1")
            self.assertEqual(bundle["scenario_count"], 1)
            self.assertEqual(len(bundle["scenarios"][0]["participants"]), 5)
            for participant in bundle["scenarios"][0]["participants"]:
                self.assertLessEqual(set(participant), set(SIMULATION_PERSONA_FIELDS))
                self.assertNotIn("representative_type", participant)
                self.assertNotIn("XXXXXXXX", participant["persona_id"])
            self.assertEqual(bundle["sampling_metadata"]["strategy"], "fixed_role_balanced_mechanism_panel")
            self.assertTrue((run_dir / "role_prototypes.json").exists())
            self.assertTrue((run_dir / "actor_classification_overlay.json").exists())
            self.assertTrue((run_dir / "panel_manifest.json").exists())
            self.assertEqual(result.handoff_metadata["sampling"]["selected_count"], 5)

    def test_refiner_can_restore_original_narratives_for_prompt_comparison(self):
        sim_data = {"scenarios": [{"participants": [{
            "representative_type": "Regulator | candidate-1",
            "professional_persona": "Previously refined text",
            "skills_and_expertise": "Contaminated domain phrase",
            "career_goals_and_ambitions": "Previously refined goal",
        }]}]}
        source_data = [{"personas": [{
            "representative_type": "Regulator | candidate-1",
            "demographics": {
                "professional_persona_pre_refine": "Original profile",
                "skills_and_expertise_pre_refine": "Original skills",
                "career_goals_and_ambitions_pre_refine": "Original goal",
            },
        }]}]

        restored, count = _restore_pre_refine_narratives(sim_data, source_data)

        participant = restored["scenarios"][0]["participants"][0]
        self.assertEqual(count, 1)
        self.assertEqual(participant["professional_persona"], "Original profile")
        self.assertEqual(participant["skills_and_expertise"], "Original skills")
        self.assertEqual(participant["career_goals_and_ambitions"], "Original goal")
        self.assertEqual(
            sim_data["scenarios"][0]["participants"][0]["skills_and_expertise"],
            "Contaminated domain phrase",
        )

    def test_mechanism_panel_does_not_run_refiner_without_explicit_opt_in(self):
        groups = [StakeholderGroup(
            stakeholder_type="patient",
            personas=[PersonaProfile(
                representative_type="Patient",
                core_objectives=["Receive protection"],
                core_constraints=["Limited information"],
            )],
        )]
        with patch("stage1.pipeline.refine_persona") as refine_mock:
            _refine_selected_personas(
                groups,
                "unused-policy-graph.json",
                {"persona": {"selection_mode": "mechanism_panel"}},
            )
        refine_mock.assert_not_called()

    def test_name_is_inferred_from_english_narrative_when_source_has_no_name_column(self):
        self.assertEqual(
            infer_persona_name({
                "professional_persona": "Curry Citizen, a retired manufacturing supervisor, mentors apprentices."
            }),
            "Curry Citizen",
        )


if __name__ == "__main__":
    unittest.main()
