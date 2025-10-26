import csv

# Path to your original CSV file
input_file_path = '/home/lydia/icarus/examples/lce-vs-probcache/cost_log.csv'
# Path to the output CSV file with the specified lines removed
output_file_path = '/home/lydia/icarus/examples/lce-vs-probcache/cleaned_log.csv'

# Read the input CSV file and write to the output CSV file
with open(input_file_path, mode='r', newline='') as infile, open(output_file_path, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Convert the rows into a list for easy manipulation
    rows = list(reader)
    skip_next = False

    # Iterate over the rows
    for i in range(len(rows)):
        if skip_next:
            # Skip this line because the previous line had only zeros and this is the "next line"
            skip_next = False
            continue

        # Check if the current row contains only 0.0 values
        if all(float(val) == 0.0 for val in rows[i]):
            # Mark the next line to be skipped
            skip_next = True
        else:
            # Write the current row to the output file
            writer.writerow(rows[i])

# Path to your CSV file
input_file_path = '/home/lydia/icarus/examples/lce-vs-probcache/cleaned_log.csv'
# Prepare to store sums of even and odd rows
even_row_sums = [0.0] * 5  # Assuming there are 5 columns
odd_row_sums = [0.0] * 5

# Read the input CSV file
with open(input_file_path, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    # Iterate through rows, determining if row is even or odd based on index
    for index, row in enumerate(reader):
        if index % 2 == 0:  # even index, impair row (starting index is 0)
            even_row_sums = [even_row_sums[i] + float(row[i]) for i in range(len(row))]
        else:  # odd index, pair row
            odd_row_sums = [odd_row_sums[i] + float(row[i]) for i in range(len(row))]

# Output the results
print("Sum of impair rows:", even_row_sums)
print("Sum of pair rows:", odd_row_sums)

# Sum of impair rows: [400.86558480000207, 0.20153089348184422, 0.0037562000000000563, 17.73058155626369, 3.0809869391700696e-05]
# Sum of pair rows: [298.3211496000018, 0.10863016621342868, 0.0024334000000001943, 1.0622633289073413, 2.4433321106098796e-06]

# "Bandwidth Cost", "Transmission Cost", "Penalty Cost", "Depreciation Cost", "Storage Cost
# >
# ESTIMATED [68.33417519999988, 0.03176995811119611, 0.00043240000000000303, 1.674207100615174, 3.7984301221288554e-06]
# REAL [102.51627000000003, 0.03733010369735005, 0.0007745999999999781, 0.7301204541275115, 2.2987560739126874e-06]
# 40, 14, 56, 78, 49

# cost is not for it
# ESTIMATED [400.86558480000207, 0.20153089348184422, 0.0037562000000000563, 17.73058155626369, 3.0809869391700696e-05]
# REAL [298.3211496000018, 0.10863016621342868, 0.0024334000000001943, 1.0622633289073413, 2.4433321106098796e-06]
# 29, 66, 42, 177, 170

# import matplotlib.pyplot as plt
# import numpy as np

# # Cost categories and their respective colors and hatch patterns
# categories = ["Bandwidth", "Transmission", "Penalty", "Depreciation", "Storage"]
# colors = ['#1F77B4', '#D62728', '#E377C2', '#FF7F0E', '#2CA02C']
# hatches = ['/', '\\', '|', '-', '+', 'x']  # Different hatch patterns for each category

# # # Estimated and real costs for demonstration
# # # estimated_costs = [68.33417519999988, 0.03176995811119611, 0.00043240000000000303, 1.674207100615174, 3.7984301221288554e-06]
# # # real_costs = [102.51627000000003, 0.03733010369735005, 0.0007745999999999781, 0.7301204541275115, 2.2987560739126874e-06]

# estimated_costs = [400.86558480000207, 0.20153089348184422, 0.0037562000000000563, 17.73058155626369, 3.0809869391700696e-05]
# real_costs = [298.3211496000018, 0.10863016621342868, 0.0024334000000001943, 1.0622633289073413, 2.4433321106098796e-06]

# # Setup for the plot
# fig, ax = plt.subplots()

# # Positions for the groups on the x-axis
# positions = np.arange(2)  # positions for 'Estimated' and 'Real'

# # Stack each category cost on the respective group
# bottom_estimated = 0
# bottom_real = 0

# for i, (color, hatch) in enumerate(zip(colors, hatches)):
#     # Add bars for estimated costs
#     ax.bar(positions[0], estimated_costs[i], color=color, hatch=hatch, width=0.4, bottom=bottom_estimated, edgecolor='black', label=categories[i])
#     bottom_estimated += estimated_costs[i]
    
#     # Add bars for real costs
#     ax.bar(positions[1], real_costs[i], color=color, hatch=hatch, width=0.4, bottom=bottom_real, edgecolor='black')
#     bottom_real += real_costs[i]

# # Add some text for labels, title and custom x-axis tick labels
# ax.set_ylabel('Costs')
# ax.set_title('Stacked Costs by Type')
# ax.set_xticks(positions)
# ax.set_xticklabels(['Estimated', 'Real'])
# ax.legend(title="Cost Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

# fig.tight_layout()

# # Save the figure
# plt.savefig('detailed_stacked_cost_comparison.jpg', format='jpg', dpi=300)  # Save as JPG file with high resolution
# plt.close(fig)  # Close the plot figure to free up memory

