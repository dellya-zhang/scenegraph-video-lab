import unittest
import numpy as np
from pipeline import extract, fixture, render, remove_object
from diffusion_adapter import graph_to_prompt


class PipelineTests(unittest.TestCase):
    def test_tracking_and_relation_reversal(self):
        graph = extract(fixture())
        self.assertTrue(all({n['id'] for n in f['nodes']} == {1,2,3} for f in graph['frames']))
        first, last = graph['frames'][0], graph['frames'][-1]
        self.assertIn({'source':1,'target':2,'relation':'left_of'},first['edges'])
        self.assertIn({'source':1,'target':2,'relation':'right_of'},last['edges'])

    def test_removal_propagates_and_preserves_original(self):
        graph = extract(fixture())
        edited = remove_object(graph, 1)
        for f in edited['frames']:
            self.assertNotIn(1, [n['id'] for n in f['nodes']])
            self.assertTrue(all(1 not in (e['source'],e['target']) for e in f['edges']))
        self.assertEqual(len(graph['frames'][0]['nodes']),3)
        recovered = extract(render(edited))
        self.assertTrue(all(len(f['nodes'])==2 for f in recovered['frames']))
        self.assertNotIn('red object', graph_to_prompt(edited))

    def test_empty_detections_and_invalid_input(self):
        self.assertEqual(extract([np.zeros((100,100,3),np.uint8)])['frames'][0]['nodes'],[])
        with self.assertRaises(ValueError):
            extract([])


if __name__ == '__main__':
    unittest.main()
