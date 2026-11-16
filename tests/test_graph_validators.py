from graph.validators import _looks_like_non_actor_policy_object


def test_direct_beneficiary_is_not_demoted_by_support_phrase() -> None:
    node = {
        "id": "YouthBeneficiary:YouthBeneficiary",
        "name": "YouthBeneficiary",
        "type": "YouthBeneficiary",
        "is_actor": True,
        "persona_eligible": True,
        "summary": "Young adults receive unlimited support for rides in some regions.",
        "attributes": {
            "agent_modality": "individual",
            "role_family": "direct_beneficiary",
        },
    }

    assert _looks_like_non_actor_policy_object(node) is False


def test_support_service_policy_object_is_still_demoted() -> None:
    node = {
        "id": "PolicySupport:OnlineSupport",
        "name": "Online Support Service",
        "type": "PolicySupport",
        "is_actor": True,
        "persona_eligible": True,
        "summary": "An online support service for applicants.",
        "attributes": {},
    }

    assert _looks_like_non_actor_policy_object(node) is True
