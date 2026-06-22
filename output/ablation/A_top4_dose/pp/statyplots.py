import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = 'generations_scored.csv'  # Replace with your actual file path
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)

    # Ensure 'coeff' is treated as numeric in case it was loaded as strings
    if 'coeff' in df.columns:
        df['coeff'] = pd.to_numeric(df['coeff'], errors='coerce')

    # Create a new column to define the three specific target groups
    df['condition'] = None
    
    # Group 1: baseline
    df.loc[df['mode'] == 'baseline', 'condition'] = 'baseline'
    
    # Group 2: ablate_head with coeff 0.5
    df.loc[(df['mode'] == 'ablate_head') & (df['coeff'] == 0.5), 'condition'] = 'ablate_head (coeff 0.5)'
    
    # Group 3: ablate_head with coeff 1.0
    df.loc[(df['mode'] == 'ablate_head') & (df['coeff'] == 1.0), 'condition'] = 'ablate_head (coeff 1.0)'

    # Drop any rows that do not fall into these three specific groups
    df = df.dropna(subset=['condition'])

    if df.empty:
        print("Warning: No rows matched the specified conditions. Check column names or values.")
        sys.exit(1)

    # Calculate the mean scores grouped by our new 'condition' column
    mean_scores = df.groupby('condition')[['coherence', 'evilness', 'adherence']].mean()

    # Explicitly enforce an orderly layout (otherwise pandas sorts them alphabetically)
    desired_order = ['baseline', 'ablate_head (coeff 0.5)', 'ablate_head (coeff 1.0)']
    # Filter order to only include groups actually found in the dataset to avoid NaN columns
    existing_order = [group for group in desired_order if group in mean_scores.index]
    mean_scores = mean_scores.reindex(existing_order)

    # Create the plot
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Transpose so metrics are on X-axis and our 3 conditions are the clustered bars
    # Increased width slightly to cleanly accommodate 3 bars per metric cluster
    mean_scores.T.plot(kind='bar', width=0.7, edgecolor='black', ax=ax)

    # Customize the chart labels and appearance
    ax.set_title('Comparison of Mean Scores by Mode and Coefficient', fontsize=14, pad=15)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Mean Values', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Condition', fontsize=10, title_fontsize=11, loc='upper right')
    
    # Adjust layout so labels or legends don't get cut off at the margins
    plt.tight_layout()

    # Save the plot to a file
    output_filename = 'mode_coeff_comparison_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Plot saved as '{output_filename}'")

    # Display the plot window
    plt.show()

if __name__ == "__main__":
    main()