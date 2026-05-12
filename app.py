from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

rating_df = pd.DataFrame({
    "Rating": [1, 2, 3, 4, 5],
    "Number of Reviews": [200000, 150000, 300000, 600000, 2500000]
})

interactive_rating_fig = px.bar(
    rating_df,
    x="Rating",
    y="Number of Reviews",
    title="Interactive Rating Distribution"
)

interactive_rating_fig.update_layout(
    xaxis_title="Star Rating",
    yaxis_title="Number of Reviews",
    template="plotly_white"
)


def fig(filename, caption):
    return html.Div(
        [
            html.Img(src=app.get_asset_url(filename), className="plot-img"),
            html.P(caption, className="caption"),
        ],
        className="figure-card",
    )


navbar = html.Nav(
    [
        dcc.Link("Home", href="/", className="nav-link"),
        dcc.Link("EDA", href="/eda", className="nav-link"),
        dcc.Link("Methods", href="/methods", className="nav-link"),
        dcc.Link("ML Models", href="/ml-models", className="nav-link"),
        dcc.Link("Findings", href="/findings", className="nav-link"),
        dcc.Link("System Design", href="/system", className="nav-link"),
    ],
    className="navbar",
)


home_layout = html.Div([
    html.Div([
        html.P("CS 163 · Group 13", className="eyebrow"),
        html.H1("Early Prediction of E-commerce Product Success"),
        html.P("Using first-4-week Amazon review signals to predict long-term product success.", className="subtitle"),
        html.P("Members: Vince Lai, Harsh Panchal"),
    ], className="hero"),

    html.H2("Project Summary"),
    html.P("""
    This project predicts whether an e-commerce product will become successful using only the first 4 weeks of post-launch review data.
    We define product success as being in the top 10% of products within the same category based on lifetime review count.
    The main goal is to help sellers and marketplace platforms identify high-potential products earlier, before months of sales history are available.
    """),

    html.Div([
        html.Div([html.H3("50,000"), html.P("Sampled Products")], className="metric-card"),
        html.Div([html.H3("Top 10%"), html.P("Success Label")], className="metric-card"),
        html.Div([html.H3("0.693"), html.P("Best ROC-AUC (Final Model)")], className="metric-card"),
        html.Div([html.H3("0.300"), html.P("Best Precision@K (Final Model)")], className="metric-card"),
    ], className="metric-grid"),

    html.H2("Main Takeaway"),
    html.P("""
    Early engagement features remain strong predictors, but adding sentiment and text-based features further improves model performance.
    The final XGBoost model achieves higher ROC-AUC and Precision@K, confirming that review content provides additional predictive signal.
    The model is most effective as a ranking system for identifying high-potential products rather than making strict binary predictions.
    """),
])


eda_layout = html.Div([
    
    html.H1("EDA and Dataset Overview"),

    html.H2("Data Source"),
    html.P("""
    We use the Amazon Customer Reviews Dataset from UCSD / McAuley Lab.
    The analysis uses 5-core category subsets from Electronics, Clothing/Shoes/Jewelry,
    Tools and Home Improvement, Toys and Games, and Sports and Outdoors.
    """),

    html.H2("Sampling Strategy"),
    html.P("""
    Since the full Amazon dataset is very large, we sampled products from selected categories.
    The modeling dataset contains products with enough review history to measure first-4-week behavior and longer-term outcomes.
    We sampled 10,000 products from each selected category, giving us 50,000 sampled products total.
    """),

    html.H2("Target Variable"),
    html.P("""
    The target variable is a binary product success label. A product is labeled successful if it is in the top 10% of its category
    by lifetime review count. This makes the task imbalanced, which is why Precision-Recall and Precision@K are important evaluation metrics.
    """),

    html.H2("EDA Insight"),
    html.P("""
    Review activity is highly uneven across products, and many products have high ratings.
    This means average rating alone is not enough to identify future success.
    Engagement and momentum features are more important.
    """),
    
    html.H2("Interactive Rating Distribution"),
    html.P("""
    This interactive chart shows that most Amazon reviews are concentrated around 5-star ratings.
    Users can hover over each bar to see the exact review count for each rating level.
    """),
    dcc.Graph(figure=interactive_rating_fig),

    fig("rating_distribution.png", "Rating Distribution: customer ratings are heavily skewed toward 5 stars."),
    fig("correlation_heatmap.png", "Feature Correlation Heatmap: review count has weak correlation with rating features, showing that popularity and rating quality are different signals."),
    fig("rating_vs_review_count.png", "Rating vs Review Count: products with more reviews do not always have higher ratings."),
])


methods_layout = html.Div([
    html.H1("Analysis Methods"),

    html.H2("Feature Engineering"),
    html.Ul([
        html.Li("Review count and review velocity in the first 4 weeks"),
        html.Li("Rating statistics and rating trends over time"),
        html.Li("Helpful vote signals (total votes and votes per review)"),
        html.Li("Text-based features including TF-IDF representations of review text"),
        html.Li("Sentiment features (average sentiment, sentiment variability, and sentiment trends)"),
        html.Li("Product category encoded as categorical features"),
    ]),


    html.H2("Models"),
    html.P("""
    We compare multiple models with increasing complexity. Logistic Regression is used as a simple baseline because it is interpretable and easy to compare.
    Random Forest is added as an ensemble model, XGBoost is used as a stronger gradient boosting model, and a Neural Network is included to test whether a more complex nonlinear model can improve prediction performance.
    """),
    html.Ul([
        html.Li("Logistic Regression: simple interpretable baseline"),
        html.Li("Random Forest: ensemble model for nonlinear relationships"),
        html.Li("XGBoost: gradient boosting model with strong predictive performance"),
        html.Li("Neural Network: more complex nonlinear model for comparison"),
    ]),

    html.H2("Text Feature Modeling"),
    html.P("""
    To incorporate review content, we used TF-IDF vectorization to extract important words and phrases from early reviews,
    and reduced dimensionality using TruncatedSVD. Using this, we found clear language differences between review bodies of high performing products versus low performing products. We also computed sentiment features such as average sentiment,
    sentiment variability, and sentiment trends over time. These features were combined with structured early signals
    to improve prediction performance.
    """),

    html.H2("Evaluation Metrics"),
    html.Ul([
        html.Li("ROC-AUC: measures general classification ability"),
        html.Li("Average Precision: useful for imbalanced data"),
        html.Li("Precision, Recall, F1: classification quality"),
        html.Li("Precision@K: business-relevant ranking metric for top predicted products"),
    ]),
])


ml_layout = html.Div([
    html.H1("ML Model Results"),

    html.H2("Model Performance"),
    html.P("""
    Both models performed better than random, but the final XGBoost model with text and sentiment features achieved the best performance.
    Compared to the baseline model, the final model improved ROC-AUC, Average Precision, and Precision@K,
    demonstrating that review text provides additional predictive signal beyond structured features.
    """),

    fig("roc_curve_4models.png", 
    "Figure 1: ROC Curve comparison across Logistic Regression, Random Forest, XGBoost, and Neural Network models. XGBoost achieved the highest ROC-AUC overall."),
    fig("pr_curve_4models.png", 
    "Figure 2: Precision-Recall Curve comparison across all four models. XGBoost and Neural Network performed better on the imbalanced success prediction task."),
    fig("metric_comparison_4models.png", 
    "Figure 3: Overall metric comparison across Logistic Regression, Random Forest, XGBoost, and Neural Network models, including ROC-AUC, Average Precision, Recall, F1, and Precision@K."),

    html.H2("Feature Importance"),
    html.P("""
    The most important features remain engagement-related, including review count, review velocity, and helpful votes.
    However, sentiment features and text-derived features also contributed to improved performance, especially in certain categories such as Toys and Games,
    indicating that how users describe products provides additional predictive information beyond ratings alone.
    """),

    fig("xgboost_importance.png", "Figure 4: XGBoost Feature Importance."),
])


findings_layout = html.Div([
    html.H1("Major Findings"),

    html.H2("Finding 1: Early Data Can Predict Product Success"),
    html.P("""
    Using only first-4-week data, the final XGBoost model achieved ROC-AUC of approximately 0.693,
    improving over the baseline model. This confirms that early product activity contains meaningful predictive signals,
    and that incorporating text and sentiment features provides additional improvements.
    """),

    html.H2("Finding 2: Engagement Features Matter More Than Ratings"),
    html.P("""
    Engagement features such as review count, review velocity, and helpful votes remain the strongest predictors.
    However, text and sentiment features also contribute to improved performance, indicating that the content of reviews
    provides additional signal beyond numerical ratings.
    """),

    html.H2("Finding 3: Review Language Reveals Category-Specific Success Patterns"),

    html.P("""
    Text analysis reveals that the language used in early reviews differs significantly between high-performing and low-performing products,
    and these patterns vary across product categories. High-performing products are consistently associated with functional,
    usage-based language, while low-performing products are associated with more generic, aesthetic, or low-information phrases.
    """),
    

    html.H3("Electronics"),
    html.P("""
    In Electronics, high-performing products are associated with terms such as "usb", "cable", "drive", "mouse", and "wireless",
    which describe concrete functionality and technical usage. These terms reflect products being actively used for specific tasks.
    In contrast, low-performing products are associated with generic phrases such as "stars", "great", and "nice",
    indicating shallow or repetitive reviews with limited technical detail.
    """),

    html.H3("Sports and Outdoors"),
    html.P("""
    In Sports and Outdoors, high-performing products are associated with practical and usage-related terms such as "water", "bottle",
    "carry", and "weight", which describe real-world use cases. Low-performing products again show more generic sentiment terms such as
    "great", "nice", and "stars", along with recreational terms like "fun" and "ball" that may reflect less specific product evaluation.
    """),

    html.H3("Tools and Home Improvement"),
    html.P("""
    For Tools and Home Improvement, high-performing products are associated with functional and performance-related terms such as
    "battery", "power", "switch", "filter", and "light", indicating that successful products are evaluated based on reliability
    and utility. Low-performing products include more generic terms such as "good", "nice", and "stars", suggesting less detailed feedback.
    """),

    html.H3("Toys and Games"),
    html.P("""
    In Toys and Games, high-performing products are associated with engagement and interaction terms such as "game", "play",
    "fun", and "toy", indicating products that are actively used and enjoyed. Low-performing products are more associated with
    aesthetic and gift-oriented language such as "cute", "figure", "gift", and "collection", suggesting novelty-based appeal
    rather than sustained engagement.
    """),

    html.H3("Clothing, Shoes, and Jewelry"),
    html.P("""
    For Clothing, Shoes, and Jewelry, high-performing products are strongly associated with fit and comfort-related terms such as
    "fit", "size", "comfortable", "wear", and "shoes". These terms indicate that successful products meet functional user needs.
    Low-performing products include more aesthetic or generic terms such as "beautiful", "watch", and "stars", suggesting less
    focus on fit and usability.
    """),

    html.P("""
    Across all categories, a consistent pattern emerges: high-performing products generate more specific, functional,
    and usage-oriented language, while low-performing products are associated with generic, repetitive, or aesthetic descriptions.
    This suggests that the quality and specificity of early review content is a strong indicator of long-term product success.
    """),

    html.H2("Finding 4: Performance Varies by Category"),
    html.P("""
    Model performance varies across product categories, indicating that early success signals are not equally informative in every domain.
    XGBoost consistently outperformed Logistic Regression across all categories, demonstrating the benefit of modeling nonlinear relationships
    and feature interactions. However, the magnitude of performance differs by category: XGBoost achieved higher accuracy in categories such as
    Tools and Home Improvement, where functionality and usage patterns are more clearly reflected in early reviews, and lower performance in
    categories like Clothing, Shoes, and Jewelry, where factors such as fit, style, and subjective preferences introduce greater variability.

    This suggests that while a global model is effective, category-specific models or feature adjustments may further improve performance
    by capturing domain-specific patterns.
    """),
    fig("correlation_features.png", "Correlation of first-4-week features with product success."),
    fig("category_auc.png", "Figure 5: Category-level ROC-AUC comparison."),
    fig("category_consistency.png", "Figure 6: Model performance consistency across categories."),

    html.H2("Text Theme Visualization"),

    html.P("""
    The following plots show the most important words and phrases associated with high-performing and low-performing products
    for each category. Positive values indicate terms more associated with successful products, while negative values indicate
    terms more associated with less successful products.
    """),

    fig("electronics_terms.png", "Electronics: functional and technical terms are associated with successful products."),
    fig("sports_terms.png", "Sports and Outdoors: usage-based terms dominate high-performing products."),
    fig("tools_terms.png", "Tools and Home Improvement: performance and utility-related terms predict success."),
    fig("toys_terms.png", "Toys and Games: engagement terms vs aesthetic/gift-based language."),
    fig("clothing_terms.png", "Clothing: fit and comfort dominate successful products."),
    
    html.H2("Additional Finding: Popularity Alone Is Not Enough"),
    html.P("""
    Instructor feedback highlighted an important limitation: a product with millions of reviews but mostly 1-star ratings should not be considered truly successful.
    This means success should not be defined only by review count. A better success definition should combine both engagement and quality, such as review volume, average rating, rating stability, and sentiment.
    """),

    html.H2("Conclusion"),
    html.P("""
    Overall, early review data provides useful predictive signals, especially when combining structured features with
    text and sentiment analysis. The results show that both engagement metrics and review language contribute to identifying
    successful products early. In particular, category-specific text patterns reveal that successful products are described
    using functional and usage-based language, while low-performing products tend to receive generic or aesthetic feedback.

    The model is best used as a ranking system to identify high-potential products rather than making perfect binary predictions.
    Future work could include richer semantic embeddings, improved success definitions, and category-specific modeling to further improve performance.
    """),
])


system_layout = html.Div([

    html.Div([
        html.H1("System Design and Deployment"),

        html.P("""
        The final website is deployed using Google App Engine. The website presents the project summary, EDA, methods,
        model results, and major findings in an interactive format.
        """, style={"marginBottom": "30px"}),

        html.H2("Current Architecture"),
        html.Ul([
            html.Li("Dash website hosted on Google App Engine"),
            html.Li("Static plots stored in the website assets folder"),
            html.Li("Modeling notebooks and code stored in GitHub"),
            html.Li("Processed results displayed through the deployed website"),
        ], style={"marginBottom": "30px"}),
        
        html.H2("Cloud Run Inference Service"),
        html.P("""
        We deployed a demo ML inference API using Google Cloud Run. The API accepts early product features,
        such as first-4-week review count, rating, review velocity, and helpful votes, then returns a predicted
        success probability.
        """),
        html.Code(
            "Cloud Run API: https://product-success-api-14643848899.us-west1.run.app",
            style={
                "display": "block",
                "backgroundColor": "#f4f4f4",
                "padding": "10px",
                "borderRadius": "6px"
            }
        ),

        html.H2("Cloud Data Storage"),
        html.Div([
            html.P("""
            We store processed dataset samples in Google Cloud Storage (GCS) to support scalable data access
            and cloud-based workflows.
            """),
            html.P(
                "Example dataset file:",
                style={"fontWeight": "bold", "marginTop": "10px"}
            ),
            html.Code(
                "gs://cs163-g13-product-data-vince/project_dataset_summary.csv",
                style={
                    "display": "block",
                    "backgroundColor": "#f4f4f4",
                    "padding": "10px",
                    "borderRadius": "6px"
                }
            )
        ], style={
            "backgroundColor": "#f9fafb",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "30px"
        }),

        html.H2("Planned Cloud Extensions"),
        html.Ul([
            html.Li("Store processed datasets in Google Cloud Storage"),
            html.Li("Deploy trained XGBoost model as a Dockerized Cloud Run inference service"),
            html.Li("Connect the website to the inference API so users can submit early product features"),
            html.Li("Improve scalability by separating website, data storage, and model inference"),
        ], style={"marginBottom": "30px"}),

        html.H2("System Design Summary"),
        html.Div([
            html.P("""
            User → App Engine Dash Website → Static Results / Future Cloud Run Model API → Cloud Storage.
            """),
            html.P("""
            This architecture separates the website, dataset, and model service,
            making the system easier to scale, maintain, and extend.
            """)
        ], style={
            "backgroundColor": "#eef6ff",
            "padding": "20px",
            "borderRadius": "10px"
        }),

    ], style={
        "maxWidth": "900px",
        "margin": "auto",
        "padding": "40px"
    })

])


app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    html.Main(id="page-content", className="page"),
])


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/eda":
        return eda_layout
    elif pathname == "/methods":
        return methods_layout
    elif pathname == "/ml-models":
        return ml_layout
    elif pathname == "/findings":
        return findings_layout
    elif pathname == "/system":
        return system_layout
    return home_layout


if __name__ == "__main__":
    app.run(debug=True)