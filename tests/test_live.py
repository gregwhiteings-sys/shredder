import os
import unittest
from app import infer

class LiveWorkflowSmokeTest(unittest.TestCase):
    @unittest.skipUnless(os.getenv('ROBOFLOW_API_KEY') and os.getenv('ROBOFLOW_SAMPLE_IMAGE'), 'Set ROBOFLOW_API_KEY and ROBOFLOW_SAMPLE_IMAGE')
    def test_published_workflow(self):
        result=infer(os.environ['ROBOFLOW_SAMPLE_IMAGE'],0,0)
        self.assertTrue({'predictions','count','measurements'}.issubset(result))
        self.assertIsInstance(result['measurements'],dict)
