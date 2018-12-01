import json
import os

from py2neo import Graph, Node, Relationship, NodeMatcher
from igraph import Graph as IGraph

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Neo4jGraph:
    def __init__(self):
        path = os.path.realpath(ROOT_DIR + '/neo4j_creds.json')
        with open(path) as f:
            data = json.load(f)
        username = data['username']
        password = data['password']
        self.graph = Graph(host="localhost", username=username, password=password)

    def add_node_by_name(self, name, node_type="Character"):
        node = Node(node_type, name=name)
        self.graph.create(node)

        return node

    def get_node_by_name(self, name):
        matcher = NodeMatcher(self.graph)
        node = matcher.match(name=name).first()

        return node

    def add_relationship(self, from_person, to_person, weight, name="INTERACTS"):
        self.graph.create(Relationship(from_person, name, to_person, weight=int(weight)))

    def add_pagerank(self):
        """
        add rankig to each node using google pagerank algorithm
        """

        query = '''
        MATCH (c1:)-[r:INTERACTS]->(c2:)
        RETURN c1.name, c2.name, r.weight AS weight
        '''
        ig = IGraph.TupleList(self.graph.run(query), weights=True)

        pg = ig.pagerank()
        pgvs = []
        for p in zip(ig.vs, pg):
            print(p)
            pgvs.append({"name": p[0]["name"], "pg": p[1]})

        write_clusters_query = '''
        UNWIND {nodes} AS n
        MATCH (c:) WHERE c.name = n.name
        SET c.pagerank = n.pg
        '''

        self.graph.run(write_clusters_query, nodes=pgvs)

    def add_communites(self):
        """
        add community membership to each node using walktrap algorithm implemented in igraph
        """

        query = '''
        MATCH (c1:)-[r:INTERACTS]->(c2:)
        RETURN c1.name, c2.name, r.weight AS weight
        '''
        ig = IGraph.TupleList(self.graph.run(query), weights=True)

        clusters = IGraph.community_walktrap(ig, weights="weight").as_clustering()

        nodes = [{"name": node["name"]} for node in ig.vs]
        for node in nodes:
            idx = ig.vs.find(name=node["name"]).index
            node["community"] = clusters.membership[idx]

        write_clusters_query = '''
        UNWIND {nodes} AS n
        MATCH (c:) WHERE c.name = n.name
        SET c.community = toInt(n.community)
        '''

        self.graph.run(write_clusters_query, nodes=nodes)


    def get_direct_neighbours(self, node):
        self.graph.match()