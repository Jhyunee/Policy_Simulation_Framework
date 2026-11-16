"""Validation boundaries for the opt-in structured ESG experiment."""
import copy
import unittest
from models.schemas import PhaseInnerSpec
from stage2_toc.esg_structured import validate_post, compatibility_payload

class StructuredESGTests(unittest.TestCase):
    def setUp(self):
        self.spec=PhaseInnerSpec(phase='Activities',prediction_fields=[dict(name='intensity',unit='qualitative',type='str',description='low medium high')],discussion_points=[])
        item=dict(statement='판단',evidence_basis='근거',uncertainty='미확인',prediction_link=dict(field='intensity',direction='uncertain',reason='조건부 영향'))
        self.data=dict(phase='Activities',persona_name='test',prediction_values={'intensity':'medium'},esg_reasoning={k:copy.deepcopy(item) for k in 'ESG'},integrated_reasoning='종합 판단',evidence=['문서'])
    def validate(self):
        return validate_post(self.data,'Activities','test',self.spec)
    def test_full_reasoning_survives_legacy_history(self):
        payload=compatibility_payload(self.validate())
        self.assertEqual(payload['prediction_values'],self.data['prediction_values'])
        for key in ('[E]','[S]','[G]','evidence_basis','uncertainty','prediction_link','종합 판단'):
            self.assertIn(key,payload['narrative'])
    def test_missing_category_rejected(self):
        del self.data['esg_reasoning']['G']
        with self.assertRaises(ValueError):self.validate()
    def test_unknown_prediction_link_rejected(self):
        self.data['esg_reasoning']['S']['prediction_link']['field']='invented'
        with self.assertRaises(ValueError):self.validate()
    def test_wrong_persona_rejected(self):
        self.data['persona_name']='other'
        with self.assertRaises(ValueError):self.validate()
    def test_no_direct_link_allowed_with_uncertainty(self):
        self.data['esg_reasoning']['S']['prediction_link']['field']=None
        self.validate()
        self.data['esg_reasoning']['S']['prediction_link']['direction']='increase'
        with self.assertRaises(ValueError):self.validate()
    def test_invented_metric_rejected(self):
        self.data['prediction_values']['esg_score']=90
        with self.assertRaises(ValueError):self.validate()

if __name__=='__main__': unittest.main()
