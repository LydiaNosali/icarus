# import networkx as nx
# import matplotlib.pyplot as plt

# # Load the GraphML file
# graphml_file = '/home/lydia/icarus/resources/topologies/Geant2012.graphml'
# G = nx.read_graphml(graphml_file)

# # Draw the graph using a layout (e.g., spring layout)
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(G)

# # Draw the nodes and edges
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10, font_weight='bold')

# # Save the plot to an image file (PNG, PDF, etc.)
# plt.savefig('graph_visualization.png')

# # Optionally, you can also save as other formats (e.g., 'graph_visualization.pdf')

# import networkx as nx
# import matplotlib.pyplot as plt

# # Load the GraphML file
# graphml_file = '/home/lydia/icarus/resources/topologies/Geant2012.graphml'
# G = nx.read_graphml(graphml_file)

# # Example events with attributes: timestamp, content, size, priority
# events = [
#     (0.8245270849967734, 364, 2206, 'low'),
#     (2.042829915730336, 203, 3947, 'low'),
#     (2.255470470783133, 259, 3162, 'high'),
#     (3.138458831958779, 1820, 3531, 'high'),
#     (4.208891335916395, 1, 3809, 'low'),
#     (5.238535190813125, 125, 2123, 'low'),
#     (6.720471805171949, 278, 3741, 'low'),
#     (7.256559129694717, 2177, 1247, 'low'),
#     (8.166047729554126, 6, 2930, 'high'),
#     (8.848358563280003, 324, 1083, 'high'),
#     (10.79267426923027, 129, 1428, 'low'),
#     (12.250304079848659, 2380, 2556, 'low')
# ]

# # Draw the base graph
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(G)  # Spring layout for node positions
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10, font_weight='bold')

# # Assign colors based on priority
# priority_colors = {'low': 'blue', 'high': 'red'}

# # Visualize events on the graph
# for timestamp, content, size, priority in events:
#     if str(content) in G:  # Assuming content corresponds to node identifiers
#         # Draw each node with size and color based on event data
#         nx.draw_networkx_nodes(G, pos, nodelist=[str(content)], node_size=size / 10, node_color=priority_colors[priority])

# # Add labels for timestamps
# event_labels = {str(content): f"t={timestamp}" for _, content, _, _ in events if str(content) in G}
# nx.draw_networkx_labels(G, pos, labels=event_labels, font_size=8, font_color='black')

# # Show or save the plot
# plt.title("Graph Visualization with Events")
# plt.savefig('graph_with_events.png')  # Save to a file if no display
# # plt.show()  # Uncomment this if running locally with a GUI


# import networkx as nx
# import plotly.graph_objects as go

# # Load the GraphML file
# graphml_file = '/home/lydia/icarus/resources/topologies/Geant2012.graphml'
# G = nx.read_graphml(graphml_file)

# # Example events with attributes: timestamp, content, size, priority
# events = [
#     (0.8245270849967734, 364, 2206, 'low'),
#     (2.042829915730336, 203, 3947, 'low'),
#     (2.255470470783133, 259, 3162, 'high'),
#     (3.138458831958779, 1820, 3531, 'high'),
#     (4.208891335916395, 1, 3809, 'low'),
#     (5.238535190813125, 125, 2123, 'low'),
#     (6.720471805171949, 278, 3741, 'low'),
#     (7.256559129694717, 2177, 1247, 'low'),
#     (8.166047729554126, 6, 2930, 'high'),
#     (8.848358563280003, 324, 1083, 'high'),
#     (10.79267426923027, 129, 1428, 'low'),
#     (12.250304079848659, 2380, 2556, 'low')
# ]
# # Get node positions using spring layout
# pos = nx.spring_layout(G)

# # Create a scatter plot for the graph nodes
# node_x = []
# node_y = []
# for node in G.nodes():
#     x, y = pos[node]
#     node_x.append(x)
#     node_y.append(y)

# # Create edge traces
# edge_x = []
# edge_y = []
# for edge in G.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_x.append(x0)
#     edge_x.append(x1)
#     edge_x.append(None)  # to draw discrete edges
#     edge_y.append(y0)
#     edge_y.append(y1)
#     edge_y.append(None)

# # Trace for edges
# edge_trace = go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# # Trace for nodes (static, updated later)
# node_trace = go.Scatter(
#     x=node_x, y=node_y,
#     mode='markers+text',
#     hoverinfo='text',
#     textposition="top center",
#     marker=dict(
#         showscale=True,
#         colorscale='YlGnBu',
#         size=10,
#         colorbar=dict(
#             thickness=15,
#             title='Node Size',
#             xanchor='left',
#             titleside='right'
#         ),
#     ))

# # Add event-based updates
# frames = []
# for timestamp, content, size, priority in events:
#     if str(content) in G:
#         node_size = [size/100 if str(n) == str(content) else 10 for n in G.nodes()]
#         node_color = ['red' if priority == 'high' and str(n) == str(content) else 'blue' if priority == 'low' else 'skyblue' for n in G.nodes()]
        
#         node_trace.marker.size = node_size
#         node_trace.marker.color = node_color

#         # Create a frame for each event
#         frames.append(go.Frame(data=[node_trace], name=f't={timestamp}', layout=go.Layout(title=f'Event at t={timestamp}')))

# # Layout for the plot
# layout = go.Layout(
#     title='Interactive Graph with Events',
#     showlegend=False,
#     hovermode='closest',
#     updatemenus=[dict(
#         type='buttons',
#         showactive=False,
#         buttons=[dict(label="Play",
#                       method="animate",
#                       args=[None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}])])],
# )

# # Create figure with initial state and frames
# fig = go.Figure(data=[edge_trace, node_trace], layout=layout, frames=frames)

# # Display the interactive figure
# fig.show()
import select
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import logging
import threading
import networkx as nx
import plotly.graph_objects as go

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        log_entry = self.format(record)
        self.text_widget.insert(tk.END, log_entry + '\n')
        self.text_widget.see(tk.END)  # Auto-scroll to the end

# Define the main application window
class IcarusGUI:
    def __init__(self, master):
        self.master = master
        master.title("ICARUS Simulation GUI")

        # Labels
        self.label = tk.Label(master, text="ICARUS Simulation Interface", font=("Helvetica", 16))
        self.label.pack(pady=10)

        # Buttons for simulation configuration
        self.file_label = tk.Label(master, text="Choose a simulation config file:")
        self.file_label.pack(pady=5)
        self.file_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.file_button.pack()

        # Display the selected file
        self.selected_file = tk.Label(master, text="", wraplength=400)
        self.selected_file.pack(pady=5)
        
        # Buttons for topology selection
        self.topology_label = tk.Label(master, text="Choose a topology (GraphML file):")
        self.topology_label.pack(pady=5)
        self.topology_button = tk.Button(master, text="Browse Topology", command=self.browse_topology)
        self.topology_button.pack()

        # Display the selected topology file
        self.selected_topology = tk.Label(master, text="", wraplength=400)
        self.selected_topology.pack(pady=5)

        # Show topology button
        self.show_topology_button = tk.Button(master, text="Show Topology", command=self.show_topology)
        self.show_topology_button.pack(pady=10)

        # Run simulation button
        self.run_button = tk.Button(master, text="Run Simulation", command=self.run_simulation_thread)
        self.run_button.pack(pady=10)

        # Output window for logs
        self.output_text = tk.Text(master, height=20, width=80)
        self.output_text.pack(pady=10)
        self.output_text.insert(tk.END, "Simulation logs will appear here...\n")

        # Set up logging to redirect to the Text widget
        self.setup_logging()

        # Initialize selected topology file variable
        self.selected_topology_file = ""

    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("main")
        logger.setLevel(logging.DEBUG)
        text_handler = TextHandler(self.output_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)

        # Configure additional loggers
        loggers_to_capture = ["main", "orchestration", "plot", "babel"]  # Add any other relevant loggers
        for logger_name in loggers_to_capture:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(text_handler)

    # Function to browse and select the config file
    def browse_file(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a config file",
                                              filetypes=(("Config files", "*.yaml *.yml *.py"), ("All files", "*.*")))
        self.selected_file.config(text=filename)

    # Function to browse and select the topology file
    def browse_topology(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a topology file",
                                              filetypes=(("GraphML files", "*.graphml"), ("All files", "*.*")))
        if filename:
            self.selected_topology.config(text=filename)
            self.selected_topology_file = filename  # Save the selected topology file path

    # Function to show the topology from the GraphML file
    def show_topology(self):
        topology_file = self.selected_topology_file
        if not topology_file:
            messagebox.showerror("Error", "Please select a topology file.")
            return

        # Load the GraphML file
        G = nx.read_graphml(topology_file)

        # Get node positions using spring layout
        pos = nx.spring_layout(G)

        # Create a scatter plot for the graph nodes
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)  # to draw discrete edges
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        # Trace for edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Trace for nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Size',
                    xanchor='left',
                    titleside='right'
                ),
            ))

        # Layout for the plot
        layout = go.Layout(
            title='Topology Visualization',
            showlegend=False,
            hovermode='closest',
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

        # Display the interactive figure
        fig.show()

    def run_simulation_thread(self):
        thread = threading.Thread(target=self.run_simulation)
        thread.start()

    # Function to run the simulation
    # Function to run the simulation
    def run_simulation(self):
        config_file = self.selected_file.cget("text")

        if not config_file:
            messagebox.showerror("Error", "Please select a configuration file.")
            return

        results = "results.pickle"
        config_file_path = os.path.abspath(config_file)
        command = ["icarus", "run", "--results", results, config_file_path]

        # Log the command that is about to be executed
        logging.getLogger("main").info("Running command: " + ' '.join(command))
        self.output_text.insert(tk.END, "Running command: " + ' '.join(command) + '\n')
        self.output_text.see(tk.END)  # Auto-scroll to the end
        self.output_text.update()  # Refresh the GUI

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

            while True:
                ready_to_read, _, _ = select.select([process.stdout, process.stderr], [], [], 1)
                for stream in ready_to_read:
                    if stream == process.stdout:
                        output = process.stdout.readline()
                        if output:
                            log_message = output.strip()
                            self.output_text.insert(tk.END, log_message + '\n')
                    elif stream == process.stderr:
                        error_message = process.stderr.readline()
                        if error_message:
                            self.output_text.insert(tk.END, error_message.strip() + '\n')

                if process.poll() is not None:
                    break

            if process.returncode == 0:
                logging.getLogger("main").info("Simulation finished successfully.")
                self.output_text.insert(tk.END, "Simulation finished successfully.\n")
            else:
                logging.getLogger("main").error("Simulation failed with return code %d." % process.returncode)
                self.output_text.insert(tk.END, "Simulation failed with return code %d.\n" % process.returncode)

        except Exception as e:
            logging.getLogger("main").error("Failed to run simulation: %s" % str(e))
            messagebox.showerror("Error", "Failed to run simulation: %s" % str(e))

# Main loop to run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    gui = IcarusGUI(root)
    root.mainloop()
