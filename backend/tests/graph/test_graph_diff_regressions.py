import unittest
from services import graph_diff_service as module
from models.architecture_models import ArchitectureGraph, ArchitectureNode, ArchitectureEdge


def graph(params=None, edge_count=0):
    return ArchitectureGraph(graph_id='g', name='g', nodes=[
        ArchitectureNode(node_id='a', kind='layer', op='nn.Linear', params=params or {}),
        ArchitectureNode(node_id='b', kind='layer', op='nn.ReLU'),
    ], edges=[ArchitectureEdge(edge_id=f'e{i}', source_node_id='a', source_port='y',
                              target_node_id='b', target_port='x') for i in range(edge_count)])


class RegressionTests(unittest.TestCase):
    def diff(self, a, b):
        return module.GraphDiffService().diff(a, b)

    def test_explicit_null_parameter_addition(self):
        result = self.diff(graph({}), graph({'device': None}))
        self.assertEqual(result.summary['modified_nodes'], 1)
        self.assertIn('device', result.nodes[0].param_changes)

    def test_explicit_null_parameter_removal(self):
        result = self.diff(graph({'device': None}), graph({}))
        self.assertEqual(result.summary['modified_nodes'], 1)
        self.assertIn('device', result.nodes[0].param_changes)

    def test_parallel_edge_removal(self):
        result = self.diff(graph(edge_count=2), graph(edge_count=1))
        self.assertEqual(result.summary['removed_edges'], 1)
        self.assertEqual(result.summary['unchanged_edges'], 1)

    def test_parallel_edge_addition(self):
        result = self.diff(graph(edge_count=1), graph(edge_count=2))
        self.assertEqual(result.summary['added_edges'], 1)
        self.assertEqual(result.summary['unchanged_edges'], 1)

    def test_parallel_edge_identity_is_bijective(self):
        result = self.diff(graph(edge_count=3), graph(edge_count=3))
        ids = [edge.edge_id_b for edge in result.edges]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result.summary['unchanged_edges'], 3)

    def test_edge_accounting_across_multiplicities(self):
        for left in range(5):
            for right in range(5):
                with self.subTest(left=left, right=right):
                    result = self.diff(graph(edge_count=left), graph(edge_count=right))
                    self.assertEqual(result.summary['unchanged_edges'], min(left, right))
                    self.assertEqual(result.summary['removed_edges'], max(0, left-right))
                    self.assertEqual(result.summary['added_edges'], max(0, right-left))

    def test_no_changes(self):
        g = graph({'device': None}, edge_count=2)
        self.assertEqual(self.diff(g, g).summary['modified_nodes'], 0)

    def test_port_rewire(self):
        a, b = graph(edge_count=1), graph(edge_count=1)
        b.edges[0].target_port = 'other'
        result = self.diff(a, b)
        self.assertEqual(result.summary['removed_edges'], 1)
        self.assertEqual(result.summary['added_edges'], 1)

    def test_rename_preserves_edges(self):
        a, b = graph(edge_count=2), graph(edge_count=2)
        b.nodes[0].node_id = 'renamed'
        for edge in b.edges:
            edge.source_node_id = 'renamed'
        result = self.diff(a, b)
        self.assertEqual(result.correspondence['a'], 'renamed')
        self.assertEqual(result.summary['unchanged_edges'], 2)


class EndpointTests(unittest.TestCase):
    def test_inline_compare_retains_null_and_edge_counts(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routers.diff import router
        from auth import get_current_user
        from types import SimpleNamespace
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(uid="diff-test")
        with TestClient(app) as client:
            response = client.post("/api/diff/compare", json={
                "a": {"graph": graph({}, 2).model_dump(mode="json")},
                "b": {"graph": graph({"device": None}, 1).model_dump(mode="json")},
            })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["summary"]["modified_nodes"], 1)
        self.assertEqual(result["summary"]["removed_edges"], 1)
        self.assertEqual(result["summary"]["unchanged_edges"], 1)
        self.assertEqual(result["nodes"][0]["param_changes"], {
            "device": {"before": None, "after": None},
        })
