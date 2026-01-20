import networkx as nx

# # Load the existing graph
# graph_path = "resources/topologies/Geant2012.graphml"
# G = nx.read_graphml(graph_path)

# # Define your carbon intensity map
# COUNTRY_CARBON_INTENSITY = {
#     "FR": 33, "DE": 224, "PL": 752, "IT": 684, "NL": 176, "BE": 143, "ES": 154,
#     "UK": 171, "SE": 36, "NO": 46, "FI": 58, "CZ": 470, "SK": 335, "HU": 294,
#     "AT": 247, "CH": 30, "PT": 83, "RO": 372, "BG": 522, "HR": 291, "SL": 329,
#     "GR": 505, "CY": 859, "IE": 329, "LU": 176, "DK": 171, "LT": 142, "LV": 152,
#     "EE": 243, "IS": 28, "RS": 372, "MD": 391, "UA": 450, "BY": 290, "TR": 466,
#     "MK": 339, "ME": 369, "MT": 475, "RU": 361, "IL": 531
# }
# DEFAULT_INTENSITY = 400

# # Add carbon intensity as a node attribute
# for node, data in G.nodes(data=True):
#     country_code = data.get("label", "").strip()
#     intensity = COUNTRY_CARBON_INTENSITY.get(country_code, DEFAULT_INTENSITY)
#     G.nodes[node]["carbon_intensity"] = intensity

# # Save updated graph to a new file
# nx.write_graphml(G, "resources/topologies/Geant2012_with_carbon.graphml")

# graph_path = "/Users/lydia/Desktop/icarus/resources/topologies/Garr201201.graphml"
# G = nx.read_graphml(graph_path)

# NODE_CARBON_INTENSITY = {
#     # 🌿 VERY_GREEN (50)
#     "Svizzera": 50, "FRA": 50, "GEANT": 50,

#     # 🍃 GREEN (120)
#     "MI-1": 120, "MI-2": 120, "MI-3": 120, "MI-4": 120,
#     "TO": 120, "TO-PIX": 120,
#     "PD": 120, "PD-2": 120,
#     "FI": 120, "Fe": 120, "Fi": 120,
#     "BO": 120, "BO-3": 120,
#     "PV": 120, "Pv": 120, "Pv-1": 120,

#     # 😐 NEUTRAL (300)
#     "TN": 300, "BR": 300, "CO": 300, "UR": 300,
#     "CZ": 300, "FUC": 300,
#     "AN": 300, "CB": 300,
#     "AQ-1": 300, "NAMEX": 300, "MIX": 300,
#     "TIX": 300, "VSIX": 300,
#     "BS": 300, "TS-1": 300, "GE": 300,
#     "ME": 300, "CT": 300, "CS": 300, "PA": 300, "PA-2": 300,

#     # 💨 DIRTY (500)
#     "RM-1": 500, "RM-2": 500, "PG": 500,
#     "MT": 500, "AQ": 500,
#     "NA": 500, "LE": 500, "FG": 500, "BA": 500, "PZ": 500,
#     "SA": 500, "Cogent": 500, "Google": 500, "CA": 500,

#     # 🔥 VERY_DIRTY (900)
#     "Level 3": 900,
#     "Global Crossing": 900,
#     "CA-1": 900,
#     "EUMED CONNECT": 900
# }


# DEFAULT_INTENSITY = 300

# for node, data in G.nodes(data=True):
#     label = data.get("label", "").strip()
#     intensity = NODE_CARBON_INTENSITY.get(label, DEFAULT_INTENSITY)
#     G.nodes[node]["carbon_intensity"] = intensity

# nx.write_graphml(G, "/Users/lydia/Desktop/icarus/resources/topologies/Garr201201_with_carbon.graphml")
# print("Saved with detailed carbon intensities.")

# import networkx as nx

# graph_path = "/Users/lydia/Desktop/icarus/resources/topologies/WideJpn.graphml"
# G = nx.read_graphml(graph_path)

# DEFAULT_INTENSITY = 300

# NODE_CARBON_INTENSITY = {
#     # 🌿 VERY_GREEN (rural / research / low-industrial load)
#     "Tsukuba": 50,
#     "Sendai": 50,
#     "Komatsu": 50,
#     "Kurashiki": 50,
#     "Sakyo": 50,

#     # 🍃 GREEN (regional cities)
#     "Hiroshima": 120,
#     "Fukuoka": 120,
#     "Nara": 120,
#     "Fujisawa": 120,
#     "Yagami": 120,

#     # 😐 NEUTRAL (mid-density metro / academic hubs)
#     "Hachioji": 300,
#     "Nezu": 300,
#     "TITECH": 300,
#     "KDDI": 300,
#     "APAN": 300,

#     # 💨 DIRTY (large metro / exchange heavy)
#     "Dojima": 500,          # Osaka area
#     "ShinKawasaki": 500,
#     "NTT Otemachi": 500,
#     "KDDI Otemachi": 500,

#     # 🔥 VERY_DIRTY (core IX / international hubs)
#     "JGN2Plus": 900,
#     "NSPIXP-3": 900,
#     "PAIX": 900,
#     "T-Lex": 900,
#     "AI3": 900,
#     "LAIIX": 900,
#     "DIX-IE": 900,

#     # 🌍 International (assume DIRTY unless modeled separately)
#     "Los Angeles": 700,
#     "San Francisco": 700,
#     "Bangkok": 700,
# }

# for node, data in G.nodes(data=True):
#     label = data.get("label", "").strip()
#     intensity = NODE_CARBON_INTENSITY.get(label, DEFAULT_INTENSITY)
#     G.nodes[node]["carbon_intensity"] = intensity

# nx.write_graphml(
#     G,
#     "/Users/lydia/Desktop/icarus/resources/topologies/WideJpn_with_carbon.graphml"
# )

# print("Saved WIDE topology with carbon intensities.")

graph_path = "resources/topologies/DeutscheTelekom.graphml"
G = nx.read_graphml(graph_path)

# Define your carbon intensity map
COUNTRY_CARBON_INTENSITY = {
    "Zurich": 57, "Geneva": 57, "Budapest": 252, "Stuttgart": 311, "Madrid": 147, "Lisbon": 127, "Milan": 365,
    "Barcelona": 147, "Paris": 32, "London": 127, "Tokyo": 501, "Chicago": 535, "Washington": 367, "Miami": 248,
    "Los Angeles": 207, "Palo Alto": 207, "San Jose": 27, "Hong Kong": 612, "Singapore": 497, "Toronto": 648, "New York": 279,
    "Frankfurt": 311, "Cologne": 311, "Hanover": 367, "Amsterdam": 220, "Ashburn": 220, "Hamburg": 311, "Dortmund": 311,
    "Dusseldorf": 311, "Vienna": 269, "Munich": 311, "Copenhagen": 155, "Stockholm": 20, "Warsaw": 797, "Moscow": 361,
    "Berlin": 311, "Leipzig": 311, "Prague": 411, "Nuremberg": 311
}
DEFAULT_INTENSITY = 400

# Add carbon intensity as a node attribute
for node, data in G.nodes(data=True):
    country_code = data.get("label", "").strip()
    intensity = COUNTRY_CARBON_INTENSITY.get(country_code, DEFAULT_INTENSITY)
    G.nodes[node]["carbon_intensity"] = intensity

# Save updated graph to a new file
nx.write_graphml(G, "resources/topologies/DeutscheTelekom_with_carbon.graphml")
