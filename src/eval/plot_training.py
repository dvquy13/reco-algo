import plotly.express as px


def plot_metric(df, index="step", col: str = None, color=None):
    # Create the plot
    cols = [index, col]
    if color:
        cols.append(color)
    fig = px.line(
        df[cols].dropna(),
        x=index,
        y=col,
        color=color,
        labels={"x": index, "y": col},
        title=f"{col} by {index}",
    )

    if color:
        fig.update_layout(showlegend=True)

    fig.show()


def plot_train_vs_val_loss(epoch_metrics_df, height=500):
    # Create the line plot
    fig = px.line(
        epoch_metrics_df,
        x="epoch",
        y="value",
        color="loss_type",
        title="train vs val loss",
        height=height,
    )

    # Loop through each trace to assign markers and labels in the same color as the lines
    for trace in fig.data:
        # Get the corresponding loss type data for the current trace
        trace_name = trace.name
        trace_df = epoch_metrics_df[epoch_metrics_df["loss_type"] == trace_name]

        # Add scatter plot with text labels in the same color as the trace line
        fig.add_scatter(
            x=trace_df["epoch"],
            y=trace_df["value"],
            mode="markers+text",
            text=trace_df["value"].apply(lambda x: f"{x:,.2f}"),
            textposition="top center",
            marker=dict(color=trace.line.color),
            textfont=dict(color=trace.line.color),
            showlegend=False,  # Disable extra legend for markers
        )

    # Display the plot
    fig.update_layout(showlegend=True)
    fig.show()
