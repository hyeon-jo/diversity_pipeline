"""
Report generation for diversity evaluation results.

Generates:
- HTML reports with visualizations
- Console summaries
- CSV exports for comparison
- Radar charts for multi-dimensional comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .evaluator import DiversityReport

logger = logging.getLogger(__name__)


def generate_html_report(
    report: DiversityReport,
    output_path: Union[str, Path],
    include_radar: bool = True
) -> str:
    """
    Generate an HTML report with visualizations.

    Args:
        report: DiversityReport to visualize.
        output_path: Path to save the HTML file.
        include_radar: Whether to include radar chart.

    Returns:
        Path to the generated HTML file.
    """
    # Prepare metrics for display
    metrics_html = ""
    score_color = _get_score_color(report.overall_diversity_score)

    # Build metric cards
    metric_cards = []

    if report.vendi_score:
        metric_cards.append(_create_metric_card(report.vendi_score))
    if report.scenario_entropy:
        metric_cards.append(_create_metric_card(report.scenario_entropy))
    if report.balance_score:
        metric_cards.append(_create_metric_card(report.balance_score))
    if report.absolute_coverage:
        metric_cards.append(_create_metric_card(report.absolute_coverage))
    if report.relative_coverage_bdd:
        metric_cards.append(_create_metric_card(report.relative_coverage_bdd))
    if report.relative_coverage_nuimages:
        metric_cards.append(_create_metric_card(report.relative_coverage_nuimages))
    if report.rarity_score:
        metric_cards.append(_create_metric_card(report.rarity_score))
    if report.intra_cluster_diversity:
        metric_cards.append(_create_metric_card(report.intra_cluster_diversity))
    if report.density_score:
        metric_cards.append(_create_metric_card(report.density_score))

    metrics_html = "\n".join(metric_cards)

    # Build radar chart data if requested
    radar_html = ""
    if include_radar:
        radar_html = _create_radar_chart(report)

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diversity Evaluation Report - {report.dataset_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --gray: #95a5a6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        .overall-score {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 30px 0;
        }}
        .score-circle {{
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: conic-gradient({score_color} {report.overall_diversity_score}%, #ecf0f1 0);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        .score-circle::before {{
            content: '';
            width: 160px;
            height: 160px;
            background: white;
            border-radius: 50%;
            position: absolute;
        }}
        .score-value {{
            position: relative;
            font-size: 3em;
            font-weight: bold;
            color: {score_color};
        }}
        .score-label {{
            position: absolute;
            bottom: -30px;
            font-size: 1.1em;
            color: #7f8c8d;
        }}
        .stats-row {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        .metric-name {{
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-bar {{
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .metric-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        .metric-description {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        .metric-interpretation {{
            color: #34495e;
            font-size: 0.85em;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 3px solid var(--primary);
        }}
        .chart-container {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }}
        .footer {{
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 0.9em;
        }}
        .reference-badge {{
            display: inline-block;
            padding: 5px 12px;
            background: #e8f4f8;
            color: #2980b9;
            border-radius: 20px;
            font-size: 0.8em;
            margin: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Diversity Evaluation Report</h1>
            <p class="subtitle">{report.dataset_name} | {report.timestamp[:10]}</p>

            <div class="overall-score">
                <div class="score-circle">
                    <span class="score-value">{report.overall_diversity_score:.0f}%</span>
                    <span class="score-label">Overall Diversity Score</span>
                </div>
            </div>

            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">{report.n_samples:,}</div>
                    <div class="stat-label">Total Samples</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{report.n_clusters}</div>
                    <div class="stat-label">Scenario Clusters</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(report.reference_datasets)}</div>
                    <div class="stat-label">Reference Datasets</div>
                </div>
            </div>

            {"".join([f'<span class="reference-badge">{ref}</span>' for ref in report.reference_datasets])}
        </div>

        {radar_html}

        <div class="metrics-grid">
            {metrics_html}
        </div>

        <div class="footer">
            Generated by Video Curation Pipeline - Diversity Metrics Module
        </div>
    </div>
</body>
</html>
"""

    # Save
    output_path = Path(output_path)
    output_path.write_text(html)
    logger.info(f"HTML report saved to {output_path}")

    return str(output_path)


def _get_score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 75:
        return "#27ae60"  # Green
    elif score >= 50:
        return "#3498db"  # Blue
    elif score >= 25:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red


def _create_metric_card(score) -> str:
    """Create HTML for a metric card."""
    color = _get_score_color(score.value)
    return f"""
    <div class="metric-card">
        <div class="metric-header">
            <div class="metric-name">{score.name}</div>
            <div class="metric-value" style="color: {color}">{score.value:.1f}{score.unit}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-bar-fill" style="width: {min(score.value, 100)}%; background: {color}"></div>
        </div>
        <div class="metric-description">{score.description}</div>
        <div class="metric-interpretation">{score.interpretation}</div>
    </div>
    """


def _create_radar_chart(report: DiversityReport) -> str:
    """Create radar chart HTML."""
    labels = []
    values = []

    if report.vendi_score:
        labels.append("Effective Diversity")
        values.append(report.vendi_score.value)
    if report.scenario_entropy:
        labels.append("Distribution Entropy")
        values.append(report.scenario_entropy.value)
    if report.balance_score:
        labels.append("Balance")
        values.append(report.balance_score.value)
    if report.absolute_coverage:
        labels.append("Coverage")
        values.append(report.absolute_coverage.value)
    if report.relative_coverage_bdd:
        labels.append("vs BDD100K")
        values.append(report.relative_coverage_bdd.value)
    if report.relative_coverage_nuimages:
        labels.append("vs nuImages")
        values.append(report.relative_coverage_nuimages.value)

    if len(labels) < 3:
        return ""

    labels_js = str(labels).replace("'", '"')
    values_js = str(values)

    return f"""
    <div class="chart-container">
        <div class="chart-title">Diversity Profile</div>
        <canvas id="radarChart" width="400" height="400"></canvas>
        <script>
            new Chart(document.getElementById('radarChart'), {{
                type: 'radar',
                data: {{
                    labels: {labels_js},
                    datasets: [{{
                        label: '{report.dataset_name}',
                        data: {values_js},
                        fill: true,
                        backgroundColor: 'rgba(52, 152, 219, 0.2)',
                        borderColor: 'rgb(52, 152, 219)',
                        pointBackgroundColor: 'rgb(52, 152, 219)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgb(52, 152, 219)'
                    }}]
                }},
                options: {{
                    elements: {{
                        line: {{ borderWidth: 3 }}
                    }},
                    scales: {{
                        r: {{
                            angleLines: {{ display: true }},
                            suggestedMin: 0,
                            suggestedMax: 100
                        }}
                    }}
                }}
            }});
        </script>
    </div>
    """


def print_report_summary(report: DiversityReport) -> None:
    """Print a formatted summary to console."""
    print("\n" + "=" * 60)
    print(f"📊 DIVERSITY EVALUATION REPORT")
    print("=" * 60)
    print(f"Dataset: {report.dataset_name}")
    print(f"Samples: {report.n_samples:,} | Clusters: {report.n_clusters}")
    print(f"References: {', '.join(report.reference_datasets) if report.reference_datasets else 'None'}")
    print("-" * 60)

    print(f"\n🎯 OVERALL DIVERSITY SCORE: {report.overall_diversity_score:.1f}%")
    print(_get_score_bar(report.overall_diversity_score))

    print("\n📈 DETAILED METRICS:")
    print("-" * 60)

    for score in [
        report.vendi_score,
        report.scenario_entropy,
        report.balance_score,
        report.absolute_coverage,
        report.relative_coverage_bdd,
        report.relative_coverage_nuimages,
        report.rarity_score,
        report.intra_cluster_diversity,
        report.density_score,
    ]:
        if score:
            print(f"\n{score.name}")
            print(f"  Value: {score.value:.1f}{score.unit}")
            print(f"  {_get_score_bar(score.value)}")
            print(f"  📝 {score.interpretation}")

    print("\n" + "=" * 60)


def _get_score_bar(value: float, width: int = 40) -> str:
    """Create a text-based progress bar."""
    filled = int(value / 100 * width)
    bar = "█" * filled + "░" * (width - filled)

    if value >= 75:
        color_start = "\033[92m"  # Green
    elif value >= 50:
        color_start = "\033[94m"  # Blue
    elif value >= 25:
        color_start = "\033[93m"  # Yellow
    else:
        color_start = "\033[91m"  # Red

    color_end = "\033[0m"

    return f"  {color_start}[{bar}]{color_end} {value:.1f}%"


def export_to_csv(
    reports: list[DiversityReport],
    output_path: Union[str, Path]
) -> str:
    """
    Export multiple reports to CSV for comparison.

    Args:
        reports: List of DiversityReports.
        output_path: Path to save CSV file.

    Returns:
        Path to saved file.
    """
    import csv

    headers = [
        "Dataset",
        "Samples",
        "Clusters",
        "Overall Score",
        "Vendi Score",
        "Entropy",
        "Balance",
        "Coverage",
        "vs BDD100K",
        "vs nuImages",
        "Rarity",
        "Intra-Cluster Div",
        "Density"
    ]

    rows = []
    for report in reports:
        row = [
            report.dataset_name,
            report.n_samples,
            report.n_clusters,
            report.overall_diversity_score,
            report.vendi_score.value if report.vendi_score else "",
            report.scenario_entropy.value if report.scenario_entropy else "",
            report.balance_score.value if report.balance_score else "",
            report.absolute_coverage.value if report.absolute_coverage else "",
            report.relative_coverage_bdd.value if report.relative_coverage_bdd else "",
            report.relative_coverage_nuimages.value if report.relative_coverage_nuimages else "",
            report.rarity_score.value if report.rarity_score else "",
            report.intra_cluster_diversity.value if report.intra_cluster_diversity else "",
            report.density_score.value if report.density_score else "",
        ]
        rows.append(row)

    output_path = Path(output_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    logger.info(f"CSV report saved to {output_path}")
    return str(output_path)


def generate_comparison_chart(
    reports: list[DiversityReport],
    output_path: Union[str, Path],
    metrics: Optional[list[str]] = None
) -> str:
    """
    Generate a comparison bar chart for multiple datasets.

    Args:
        reports: List of DiversityReports to compare.
        output_path: Path to save HTML file.
        metrics: List of metrics to include. If None, uses all available.

    Returns:
        Path to saved file.
    """
    if metrics is None:
        metrics = ["overall", "vendi", "entropy", "balance", "coverage"]

    datasets = [r.dataset_name for r in reports]
    data = {m: [] for m in metrics}

    for report in reports:
        data["overall"].append(report.overall_diversity_score) if "overall" in metrics else None
        data["vendi"].append(report.vendi_score.value if report.vendi_score else 0) if "vendi" in metrics else None
        data["entropy"].append(report.scenario_entropy.value if report.scenario_entropy else 0) if "entropy" in metrics else None
        data["balance"].append(report.balance_score.value if report.balance_score else 0) if "balance" in metrics else None
        data["coverage"].append(report.absolute_coverage.value if report.absolute_coverage else 0) if "coverage" in metrics else None

    # Generate HTML with Chart.js
    datasets_js = str(datasets).replace("'", '"')

    chart_datasets = []
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (metric, values) in enumerate(data.items()):
        if values:
            chart_datasets.append({
                "label": metric.capitalize(),
                "data": values,
                "backgroundColor": colors[i % len(colors)],
            })

    datasets_config = str(chart_datasets).replace("'", '"')

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ text-align: center; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dataset Diversity Comparison</h1>
        <canvas id="comparisonChart"></canvas>
    </div>
    <script>
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'bar',
            data: {{
                labels: {datasets_js},
                datasets: {datasets_config}
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        title: {{ display: true, text: 'Score (%)' }}
                    }}
                }},
                plugins: {{
                    legend: {{ position: 'top' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html)
    logger.info(f"Comparison chart saved to {output_path}")

    return str(output_path)
