from dash import Dash, html, dcc, Input, Output

app = Dash(__name__)
server = app.server


def fig(filename, caption):
    return html.Div([
        html.Img(src=app.get_asset_url(filename), className="figure-img"),
        html.P(caption, className="caption")
    ], className="figure-card")


def metric_card(value, label):
    return html.Div([
        html.H3(value),
        html.P(label)
    ], className="metric-card")


home_layout = html.Div([
    html.Div([
        html.P("CS 163 · Group 13", className="eyebrow"),
        html.H1("Early Prediction of E-commerce Product Success"),
        html.P("Using first-4-week Amazon review signals to predict long-term product success.",
               className="subtitle"),
        html.P("Members: Vince Lai, Harsh Panchal", className="members")
    ], className="hero"),

    html.H2("Project Summary"),
    html.P("""
    This project predicts whether an e-commerce product will become successful using only the first
    4 weeks of post-launch review data. We define product success as being in the top 10% of products
    within the same category based on lifetime review count. The main goal is to help sellers and
    marketplace platforms identify high-potential products earlier, before months of sales history
    are available.
    """),

    html.Div([
        metric_card("50,000", "Sampled Products"),
        metric_card("Top 10%", "Success Label"),
        metric_card("0.679", "Best ROC-AUC"),
        metric_card("0.287", "Best Precision@K")
    ], className="metric-grid"),

    html.H2("Main Takeaway"),
    html.P("""
    Early engagement features such as review count, votes, and review velocity are more useful than
    ratings alone. XGBoost performs better than Logistic Regression, but the model is best used as a
    ranking tool rather than a perfect yes/no classifier.
    """)
])


eda_layout = html.Div([
    html.H2("EDA and Dataset Overview"),

    html.H3("Data Source"),
    html.P("""
    We use the Amazon Customer Reviews Dataset from UCSD / McAuley Lab. The analysis uses 5-core
    category subsets from Electronics, Clothing/Shoes/Jewelry, Tools and Home Improvement,
    Toys and Games, and Sports and Outdoors.
    """),

    html.H3("Sampling Strategy"),
    html.P("""
    Since the full Amazon dataset is very large, we sampled products from selected categories.
    The modeling dataset contains products with enough review history to measure first-4-week
    behavior and longer-term outcomes.
    """),

    html.H3("Target Variable"),
    html.P("""
    The target variable is a binary product success label. A product is labeled successful if it is
    in the top 10% of its category by lifetime review count. This makes the task imbalanced, which is
    why Precision-Recall and Precision@K are important evaluation metrics.
    """),

    html.H3("EDA Insight"),
    html.P("""
    Review activity is highly uneven across products, and many products have high ratings. This
    means average rating alone is not enough to identify future success. Engagement and momentum
    features are more important.
    """),

    fig("correlation_features.png", "EDA Result: Correlation of first-4-week features with product success.")
])


methods_layout = html.Div([
    html.H2("Analysis Methods"),

    html.H3("Feature Engineering"),
    html.Ul([
        html.Li("Review count in the first 4 weeks"),
        html.Li("Reviews per day / review velocity"),
        html.Li("Mean, median, min, max, and standard deviation of ratings"),
        html.Li("Total helpful votes and votes per review"),
        html.Li("Review text length and summary length"),
        html.Li("Product category encoded as categorical features")
    ]),

    html.H3("Models"),
    html.P("""
    We compare a simple baseline model and a stronger tree-based model. Logistic Regression is used
    as the baseline because it is interpretable and easy to compare. XGBoost is used because it can
    capture nonlinear relationships and feature interactions.
    """),

    html.H3("Evaluation Metrics"),
    html.Ul([
        html.Li("ROC-AUC: measures general classification ability"),
        html.Li("Average Precision: useful for imbalanced data"),
        html.Li("Precision, Recall, F1: classification quality"),
        html.Li("Precision@K: business-relevant ranking metric for top predicted products")
    ])
])


models_layout = html.Div([
    html.H2("ML Model Results"),

    html.H3("Model Performance"),
    html.P("""
    Both models performed better than random. XGBoost outperformed Logistic Regression across all
    major metrics, showing that early product review signals contain useful predictive information.
    """),

    fig("roc_curve.png", "Figure 1: ROC Curve Comparison. XGBoost achieves higher ROC-AUC than Logistic Regression."),
    fig("pr_curve.png", "Figure 2: Precision-Recall Curve. XGBoost performs better on the imbalanced success label."),
    fig("metric_comparison.png", "Figure 3: Metric Comparison. XGBoost improves ROC-AUC, recall, F1, and Precision@K."),

    html.H3("Feature Importance"),
    html.P("""
    The most important features are engagement-related: total votes, votes per review, review count,
    and review velocity. Rating features are less useful because ratings are highly concentrated near
    4 and 5 stars.
    """),

    fig("xgboost_importance.png", "Figure 4: XGBoost Feature Importance.")
])


findings_layout = html.Div([
    html.H2("Major Findings"),

    html.H3("Finding 1: Early Data Can Predict Product Success"),
    html.P("""
    Using only first-4-week data, XGBoost achieved ROC-AUC around 0.679 compared to 0.638 for
    Logistic Regression. This supports the idea that early product activity contains predictive
    signals.
    """),

    html.H3("Finding 2: Engagement Features Matter More Than Ratings"),
    html.P("""
    Features such as review count, total votes, votes per review, and review velocity were stronger
    signals than average rating. This suggests success is more connected to early engagement and
    momentum than star rating alone.
    """),

    html.H3("Finding 3: Performance Varies by Category"),
    html.P("""
    Model performance is not equal across categories. XGBoost performed best for Tools and Home
    Improvement and weaker for Clothing, Shoes, and Jewelry. This suggests category-specific models
    may improve future performance.
    """),

    fig("category_auc.png", "Figure 5: Category-level ROC-AUC comparison."),
    fig("category_consistency.png", "Figure 6: Model performance consistency across categories."),

    html.H2("Conclusion"),
    html.P("""
    Overall, early review data provides useful but incomplete signals. The best use case is ranking
    high-potential products rather than making perfect binary predictions. Future work should add NLP
    features such as sentiment and review topics, test different early time windows, and build
    category-specific models.
    """)
])


system_layout = html.Div([
    html.H2("System Design and Deployment"),

    html.P("""
    The final website is deployed using Google App Engine. The website presents the project summary,
    EDA, methods, model results, and major findings in an interactive format.
    """),

    html.H3("Current Architecture"),
    html.Ul([
        html.Li("Dash website hosted on Google App Engine"),
        html.Li("Static plots stored in the website assets folder"),
        html.Li("Modeling notebooks and code stored in GitHub"),
        html.Li("Processed results displayed through the deployed website")
    ]),

    html.H3("Planned Cloud Extensions"),
    html.Ul([
        html.Li("Store processed datasets in Google Cloud Storage"),
        html.Li("Deploy trained XGBoost model as a Dockerized Cloud Run inference service"),
        html.Li("Connect the website to the inference API so users can submit early product features"),
        html.Li("Improve scalability by separating website, data storage, and model inference")
    ]),

    html.H3("System Design Summary"),
    html.P("""
    User → App Engine Dash Website → Static Results / Future Cloud Run Model API → Cloud Storage.
    This separates the website, dataset, and model service, making the project easier to maintain
    and extend.
    """)
])


app.layout = html.Div([
    html.Div([
        html.A("Home", href="#home", className="nav-link"),
        html.A("EDA", href="#eda", className="nav-link"),
        html.A("Methods", href="#methods", className="nav-link"),
        html.A("ML Models", href="#models", className="nav-link"),
        html.A("Findings", href="#findings", className="nav-link"),
        html.A("System Design", href="#system", className="nav-link")
    ], className="navbar"),

    html.Div([
        html.Section(home_layout, id="home", className="section"),
        html.Section(eda_layout, id="eda", className="section"),
        html.Section(methods_layout, id="methods", className="section"),
        html.Section(models_layout, id="models", className="section"),
        html.Section(findings_layout, id="findings", className="section"),
        html.Section(system_layout, id="system", className="section")
    ], className="container")
])


if __name__ == "__main__":
    app.run(debug=True)