import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = 'generations_scored.csv'
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)

    # Calculate the mean scores grouped by 'mode'
    mean_scores = df.groupby('mode')[['coherence', 'evilness', 'adherence']].mean()

    # Create the plot
    # 'figsize' ensures the window is big enough to look clean outside of Jupyter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Transpose the dataframe so metrics are on the X-axis and modes are the bars
    mean_scores.T.plot(kind='bar', width=0.6, edgecolor='black', ax=ax)

    # Customize the chart labels and appearance
    ax.set_title('Comparison of Mean Scores by Mode', fontsize=14, pad=15)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Mean Values', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Mode', fontsize=11, title_fontsize=12)
    
    # Adjust layout so labels don't get cut off
    plt.tight_layout()

    # OPTION 1: Save the plot to a file (highly recommended for scripts)
    output_filename = 'mode_comparison_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Plot saved as '{output_filename}'")

    # OPTION 2: Pop up a window displaying the plot
    # Note: This will pause script execution until you close the window
    plt.show()

if __name__ == "__main__":
    main()