import networkx as nx
import matplotlib.pyplot as plt

# Load the graph
# graph_path = '/Users/lydia/Desktop/icarus/resources/topologies/Geant2012.graphml'
graph_path = '/Users/lydia/Desktop/icarus/resources/topologies/DeutscheTelekom.graphml'
G = nx.read_graphml(graph_path)

# Generate positions for each node using a layout
pos = nx.spring_layout(G)  # Can use other layouts like nx.kamada_kawai_layout(G) if preferred

# Define edge_labels by checking if the 'LinkLabel' attribute exists or use a default labeling
edge_labels = {}
for edge in G.edges(data=True):
    label = edge[2].get('LinkLabel', f"{edge[0]}-{edge[1]}")  # Use 'LinkLabel' if available, else default
    edge_labels[(edge[0], edge[1])] = label

# Draw the graph
plt.figure(figsize=(14, 10))  # Larger figure size for clarity

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
nx.draw_networkx_edges(G, pos, edge_color='gray')

# Node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# Edge labels: only display if the edge has a significant label
significant_edges = {k: v for k, v in edge_labels.items() if 'Gbps' in v}  # Example filter for clarity
nx.draw_networkx_edge_labels(G, pos, edge_labels=significant_edges, font_color='red', font_size=10)

# Save the figure
output_path = '/Users/lydia/Desktop/icarus/resources/topologies/DeutscheTelekom_Graph_Visualization.png'
# plt.title('Improved Graph Visualization with Transmission Capacity')
plt.axis('off')  # Turn off the axis
plt.savefig(output_path)
plt.close()

print(f"Graph saved as {output_path}")
