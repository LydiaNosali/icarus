import networkx as nx

# Load the existing graph
# graph_path = "resources/topologies/Geant2012.graphml"
# G = nx.read_graphml(graph_path)

# # Define your carbon intensity map
# COUNTRY_CARBON_INTENSITY = {
#     "FR": 50, "DE": 400, "PL": 600, "IT": 300, "NL": 300, "BE": 180, "ES": 250,
#     "UK": 200, "SE": 40, "NO": 20, "FI": 80, "CZ": 420, "SK": 250, "HU": 300,
#     "AT": 100, "CH": 20, "PT": 200, "RO": 350, "BG": 380, "HR": 250, "SI": 150,
#     "GR": 270, "CY": 500, "IE": 250, "LU": 60, "DK": 120, "LT": 100, "LV": 100,
#     "EE": 500, "IS": 0, "RS": 400, "MD": 400, "UA": 450, "BY": 500, "TR": 450,
#     "MK": 450, "ME": 400, "MT": 350, "RU": 450, "IL": 430
# }
# DEFAULT_INTENSITY = 400

# # Add carbon intensity as a node attribute
# for node, data in G.nodes(data=True):
#     country_code = data.get("label", "").strip()
#     intensity = COUNTRY_CARBON_INTENSITY.get(country_code, DEFAULT_INTENSITY)
#     G.nodes[node]["carbon_intensity"] = intensity

# # Save updated graph to a new file
# nx.write_graphml(G, "resources/topologies/Geant2012_with_carbon.graphml")
graph_path = "resources/topologies/Garr201201.graphml"
G = nx.read_graphml(graph_path)
NODE_CARBON_INTENSITY = {
    "MI-1": 280, "MI-2": 280, "MI-3": 280, "MI-4": 280,
    "TO": 285, "TO-PIX": 285,
    "PD": 290, "PD-2": 290, "VE": 290, "TN": 280, "BR": 290,
    "FI": 285, "Fe": 285, "BO": 285, "BO-3": 285,
    "PI": 290, "CO": 285, "Pv": 285, "Pv-1": 285,
    "UR": 285, "CZ": 295, "FUC": 285,
    "RM-1": 300, "RM-2": 300, "PG": 300, "AN": 300, "CB": 300,
    "MT": 300, "AQ": 300, "AQ-1": 300,
    "NA": 320, "LE": 320, "FG": 320, "BA": 320, "PZ": 320,
    "SA": 320, "ME": 325, "CT": 325, "CS": 325, "PA": 325, "PA-2": 325,
    "SS": 330, "TIX": 295, "VSIX": 285,
    "Svizzera": 20, "FRA": 50, "GEANT": 100,
    "EUMED CONNECT": 450, "Level 3": 400, "Cogent": 400, "Google": 350,
    "Global Crossing": 400, "NAMEX": 300, "MIX": 300,
    "CA": 400, "CA-1": 400,
    "BS": 300, "Fi": 285, "TS-1": 295, "GE": 290
}

DEFAULT_INTENSITY = 300

for node, data in G.nodes(data=True):
    label = data.get("label", "").strip()
    intensity = NODE_CARBON_INTENSITY.get(label, DEFAULT_INTENSITY)
    G.nodes[node]["carbon_intensity"] = intensity

nx.write_graphml(G, "resources/topologies/Garr201201_with_carbon.graphml")
print("Saved with detailed carbon intensities.")
