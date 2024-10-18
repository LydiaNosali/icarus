import csv

# Input and output file paths
request_file = "examples/lce-vs-probcache/traces/events.csv"  # Your request trace file (CSV format)
contents_file = "examples/lce-vs-probcache/traces/events_contents.csv"  # The unique contents file to create (CSV format)

# Dictionary to store unique content with size
content_dict = {}

# Read the request file and extract unique content IDs and their sizes
with open(request_file, 'r') as req_file:
    reader = csv.reader(req_file)
    next(reader)  # Skip the header
    for row in reader:
        timestamp, content_id, size, priority = row
        size = int(size)  # Convert size to integer
        priority = str(priority)
        
        # If the content_id is not in the dictionary, add it
        if content_id not in content_dict:
            content_dict[content_id] = (size, priority)

# Write the unique content identifiers and sizes to the contents file
print(content_dict.__len__())
with open(contents_file, 'w', newline='') as cont_file:
    writer = csv.writer(cont_file)
    
    # Write header for the contents file
    writer.writerow(["content_id", "size", "priority"])
    
    for content_id, (size, priority) in content_dict.items():
        # Here you can assign a priority value (setting a default of 1)
        writer.writerow([content_id, size, priority])

print(f"Unique contents file '{contents_file}' generated successfully.")
