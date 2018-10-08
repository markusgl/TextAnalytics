import json
from py2neo import Graph, Node, Relationship, NodeMatcher


class Neo4jGraph:
    def __init__(self, path='neo4j_creds.json'):
        with open(path) as f:
            data = json.load(f)
        username = data['username']
        password = data['password']
        self.graph = Graph(host="localhost", username=username, password=password)

    def add_node_by_name(self, name):
        node = Node("Person", name=name)
        self.graph.create(node)

        return node

    def get_node_by_name(self, name):
        matcher = NodeMatcher(self.graph)
        node = matcher.match('Person', name=name).first()

        return node

    def add_relationship(self, from_person, to_person):
        self.graph.create(Relationship(from_person, "relatedto", to_person))
