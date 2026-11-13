import io
import unittest
from unittest.mock import patch
from PIL import Image
from app import app, JOBS

class AppTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
        JOBS.clear()

    def test_review_gate_and_exports(self):
        image=Image.new('RGB',(120,80),'white'); out=io.BytesIO();image.save(out,'JPEG');out.seek(0)
        fake={'count':2,'predictions':{'predictions':[]},'measurements':{'count':2,'measured_count':1,'pieces':[{'id':1,'area_px2':125,'area_cm2':None}]}}
        with patch('app.infer',return_value=fake):
            res=self.client.post('/api/analyze',data={'image':(out,'test.jpg'),'reference_distance_px':'0','reference_distance_cm':'0'},content_type='multipart/form-data')
        self.assertEqual(res.status_code,200)
        data=res.get_json();self.assertEqual(data['count'],2);self.assertEqual(len(data['pieces']),1)
        id=data['id'];self.assertEqual(self.client.get(f'/api/jobs/{id}/export.pdf').status_code,403)
        res=self.client.post(f'/api/jobs/{id}/review',json={'confirm':True,'reviewer':'Inspector','reviewed_count':1,'excluded_ids':[]})
        self.assertEqual(res.status_code,200)
        self.assertIn(b'Human reviewed',self.client.get(f'/api/jobs/{id}/export.csv').data)
        self.assertTrue(self.client.get(f'/api/jobs/{id}/export.pdf').data.startswith(b'%PDF'))

if __name__=='__main__': unittest.main()
