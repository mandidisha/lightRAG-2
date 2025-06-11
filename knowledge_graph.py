import networkx as nx

class KnowledgeGraph1:
    """
    A simple knowledge graph using a directed multigraph structure (NetworkX).
    Nodes represent entities, and edges represent relations (triples) between entities.
    """
    def __init__(self):
        # Use a MultiDiGraph to allow multiple edges (different relations) between the same nodes
        self.graph = nx.MultiDiGraph()
    
    def add_entity(self, entity_name):
        """Add an entity (node) to the graph if not already present."""
        if entity_name not in self.graph.nodes:
            self.graph.add_node(entity_name)
    
    def add_triple(self, subject, relation, obj):
        """Add a triple (subject -[relation]-> object) to the graph."""
        self.add_entity(subject)
        self.add_entity(obj)
        self.graph.add_edge(subject, obj, relation=relation)
    
    def add_edge(self, subject, relation, obj):
        """Alias for add_triple, for compatibility with some scripts."""
        self.add_triple(subject, relation, obj)
    
    def find_node(self, name_substring):
        """Find nodes whose name contains the given substring (case-insensitive)."""
        matches = []
        for node in self.graph.nodes():
            if name_substring.lower() in str(node).lower():
                matches.append(node)
        return matches
    
    def find_edges(self, node=None, relation=None):
        """Retrieve edges (triples) from the graph, optionally filtered by node or relation."""
        results = []
        for u, v, data in self.graph.edges(data=True):
            rel = data.get('relation')
            if relation and rel != relation:
                continue
            if node:
                if u == node or v == node:
                    results.append((u, rel, v))
            else:
                results.append((u, rel, v))
        return results
    
    def number_of_nodes(self):
        """Return the count of nodes (entities) in the graph."""
        return self.graph.number_of_nodes()
    
    def number_of_edges(self):
        """Return the count of edges (relations) in the graph."""
        return self.graph.number_of_edges()
