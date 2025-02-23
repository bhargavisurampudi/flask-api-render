import networkx as nx
import matplotlib.pyplot as plt

# Create a graph representing transit routes
G = nx.Graph()
edges = [
    ('A', 'B', 5), ('A', 'C', 10), ('B', 'C', 3),
    ('B', 'D', 7), ('C', 'D', 2), ('D', 'E', 8), ('C', 'E', 6)
]
G.add_weighted_edges_from(edges)

# Find the shortest path
shortest_path = nx.shortest_path(G, source='A', target='E', weight='weight')
print(f"🚀 Optimized Route: {shortest_path}")

# Visualize the route
plt.figure(figsize=(6, 4))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)
plt.title("Optimized Transit Routes")
plt.show()
