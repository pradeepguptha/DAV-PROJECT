# advanced_sales_dashboard.py
import base64
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression

# -------------------------
# Helpers
# -------------------------
def try_parse_date_series(s):
    try:
        parsed = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
        if parsed.notna().sum() >= max(1, len(parsed) * 0.1):  # at least 10% parse success
            return parsed
    except Exception:
        pass
    return None

def detect_columns(df):
    # find first likely date column
    date_col = None
    for c in df.columns:
        parsed = try_parse_date_series(df[c])
        if parsed is not None:
            df[c] = parsed
            date_col = c
            break

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def guess(keywords):
        return next((c for c in df.columns if any(k in c.lower() for k in keywords)), None)

    sales_col = guess(["sales", "amount", "revenue", "total", "value", "price"])
    profit_col = guess(["profit", "margin", "gain"])
    product_col = guess(["product", "item", "sku", "category", "name"])
    region_col = guess(["region", "state", "city", "country", "area"])

    return {
        "date_col": date_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "sales_col": sales_col,
        "profit_col": profit_col,
        "product_col": product_col,
        "region_col": region_col,
    }

def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    # try CSV then Excel
    try:
        return pd.read_csv(io.StringIO(decoded.decode("utf-8")), low_memory=False)
    except Exception:
        try:
            return pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            raise e

# -------------------------
# App setup
# -------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

TITLE_STYLE = {
    "textAlign": "center",
    "color": "#00e1ff",
    "fontWeight": "bold",
    "fontSize": "30px",
    "marginBottom": "12px"
}

app.layout = dbc.Container([
    html.H2("🚀 Advanced Sales Analytics Dashboard", style=TITLE_STYLE),

    dcc.Upload(
        id="upload-data",
        children=html.Div(["📂 Drag & Drop or Click to Upload CSV / Excel"]),
        style={
            "width": "100%", "height": "80px", "lineHeight": "80px",
            "borderWidth": "2px", "borderStyle": "dashed",
            "borderRadius": "10px", "textAlign": "center",
            "cursor": "pointer", "backgroundColor": "#111"
        },
        multiple=False,
        accept=".csv,.xlsx,.xls"
    ),

    html.Br(),

    dash_table.DataTable(id="data-preview", page_size=5, style_table={"overflowX": "auto"}),

    html.Br(),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Filters", style={"color": "#00e1ff"}),
                    html.Div(id="detected-fields", style={"fontSize": "13px", "color": "#bbb"}),
                    html.Br(),
                    dbc.Label("Date range:"),
                    dcc.DatePickerRange(id="date-range"),
                    html.Br(), html.Br(),
                    dbc.Label("Select Products (multi):"),
                    dcc.Dropdown(id="filter-product", multi=True, placeholder="Select product(s)"),
                    html.Br(),
                    dbc.Label("Select Regions (multi):"),
                    dcc.Dropdown(id="filter-region", multi=True, placeholder="Select region(s)"),
                    html.Br(),
                    dbc.Label("Select Chart Type(s):"),
                    dcc.Dropdown(
                        id="chart-type",
                        multi=True,
                        value=["line"],
                        options=[
                            {"label": "Line (time series)", "value": "line"},
                            {"label": "Bar (aggregated)", "value": "bar"},
                            {"label": "Histogram", "value": "hist"},
                            {"label": "Boxplot", "value": "box"},
                            {"label": "Pie (categorical)", "value": "pie"},
                            {"label": "Scatter (numeric)", "value": "scatter"},
                            {"label": "Correlation Heatmap", "value": "corr"},
                        ]
                    ),
                    html.Br(),
                    dbc.Button("Generate Charts", id="btn-generate", color="success", className="me-2"),
                    dbc.Button("Forecast (30)", id="btn-forecast", color="warning"),
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Sales"), html.H3(id="kpi-sales")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Profit"), html.H3(id="kpi-profit")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Avg Order Value"), html.H3(id="kpi-aov")]))),
            ], className="mb-3"),
            dbc.Card(dbc.CardBody([
                html.H4("📊 Generated Visualizations", style={"color": "#00e1ff"}),
                html.Div(id="chart-grid")
            ]))
        ], width=8)
    ]),

    html.Br(),
    html.H4("Insights", style={"color": "#00e1ff"}),
    html.Pre(id="insights", style={"whiteSpace": "pre-wrap", "fontSize": "15px", "color": "#ddd"}),

    dcc.Store(id="stored-data"),
    dcc.Store(id="detections"),
], fluid=True)

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("stored-data", "data"),
    Output("data-preview", "data"),
    Output("data-preview", "columns"),
    Output("detections", "data"),
    Input("upload-data", "contents")
)
def load_file(contents):
    if not contents:
        return None, [], [], {}
    df = parse_contents(contents)
    det = detect_columns(df)
    preview = df.head(10).to_dict("records")
    cols = [{"name": c, "id": c} for c in df.columns]
    return df.to_json(date_format="iso", orient="split"), preview, cols, det

@app.callback(
    Output("detected-fields", "children"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Output("filter-product", "options"),
    Output("filter-region", "options"),
    Input("detections", "data"),
    State("stored-data", "data")
)
def setup_filters(detections, stored):
    if not stored or not detections:
        return "Upload a dataset to see detected columns.", None, None, [], []
    df = pd.read_json(stored, orient="split")
    det = detections
    parts = []
    if det.get("date_col"):
        parts.append(f"Date: {det['date_col']}")
        s = df[det["date_col"]].dropna()
        start = s.min().date().isoformat() if not s.empty else None
        end = s.max().date().isoformat() if not s.empty else None
    else:
        start = end = None
    prod_opts = []
    if det.get("product_col") and det["product_col"] in df.columns:
        prod_opts = [{"label": str(v), "value": v} for v in df[det["product_col"]].dropna().unique()]
    reg_opts = []
    if det.get("region_col") and det["region_col"] in df.columns:
        reg_opts = [{"label": str(v), "value": v} for v in df[det["region_col"]].dropna().unique()]
    if det.get("sales_col"):
        parts.append(f"Sales: {det['sales_col']}")
    if det.get("profit_col"):
        parts.append(f"Profit: {det['profit_col']}")
    return " | ".join(parts), start, end, prod_opts, reg_opts

@app.callback(
    Output("chart-grid", "children"),
    Output("kpi-sales", "children"),
    Output("kpi-profit", "children"),
    Output("kpi-aov", "children"),
    Output("insights", "children"),
    Input("btn-generate", "n_clicks"),
    Input("btn-forecast", "n_clicks"),
    State("stored-data", "data"),
    State("detections", "data"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("filter-product", "value"),
    State("filter-region", "value"),
    State("chart-type", "value")
)
def generate_and_forecast(gen_click, forecast_click, stored, detections,
                          start_date, end_date, product_filter, region_filter, chart_types):
    # default empty outputs
    empty_fig = dcc.Graph(figure=go.Figure())
    if not stored or not detections:
        return [html.Div("Upload a dataset and click Generate.")], "-", "-", "-", "Upload a dataset first."

    df = pd.read_json(stored, orient="split")
    det = detections
    date_col = det.get("date_col")
    sales_col = det.get("sales_col")
    profit_col = det.get("profit_col")
    product_col = det.get("product_col")
    region_col = det.get("region_col")

    # determine which button triggered
    trig = callback_context.triggered
    if not trig:
        return [html.Div("Select charts and click Generate.")], "-", "-", "-", "No action yet."
    clicked = trig[0]["prop_id"].split(".")[0]

    # filtering
    dff = df.copy()
    try:
        if date_col and start_date:
            dff = dff[dff[date_col] >= pd.to_datetime(start_date)]
        if date_col and end_date:
            dff = dff[dff[date_col] <= pd.to_datetime(end_date)]
    except Exception:
        pass

    # multi-select filters
    if product_col and product_filter:
        if isinstance(product_filter, list):
            dff = dff[dff[product_col].isin(product_filter)]
        else:
            dff = dff[dff[product_col] == product_filter]
    if region_col and region_filter:
        if isinstance(region_filter, list):
            dff = dff[dff[region_col].isin(region_filter)]
        else:
            dff = dff[dff[region_col] == region_filter]

    # KPIs
    try:
        total_sales = f"{dff[sales_col].dropna().astype(float).sum():,.2f}" if sales_col and sales_col in dff.columns else "-"
    except Exception:
        total_sales = "-"
    try:
        total_profit = f"{dff[profit_col].dropna().astype(float).sum():,.2f}" if profit_col and profit_col in dff.columns else "-"
    except Exception:
        total_profit = "-"
    try:
        aov = f"{dff[sales_col].dropna().astype(float).mean():,.2f}" if sales_col and sales_col in dff.columns else "-"
    except Exception:
        aov = "-"

    insights = []

    # If forecast button pressed -> produce forecast figure (simple linear regression on aggregated sales)
    if clicked == "btn-forecast":
        if not date_col or not sales_col:
            return [html.Div("Need date and sales column for forecast.")], total_sales, total_profit, aov, "Forecast requires date & sales columns."
        agg = dff.groupby(date_col).agg({sales_col: "sum"}).reset_index().sort_values(date_col)
        if agg.shape[0] < 3:
            return [html.Div("Not enough aggregated points to forecast.")], total_sales, total_profit, aov, "Need at least 3 aggregated dates."
        agg = agg.reset_index(drop=True)
        agg["t"] = np.arange(len(agg))
        model = LinearRegression()
        model.fit(agg[["t"]], agg[sales_col].astype(float).values)
        N = 30
        future_t = np.arange(len(agg), len(agg) + N)
        y_pred = model.predict(future_t.reshape(-1, 1))
        median_delta = agg[date_col].diff().median() or pd.Timedelta(days=1)
        future_dates = [agg[date_col].max() + (i + 1) * median_delta for i in range(N)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg[date_col], y=agg[sales_col], mode="lines+markers", name="History"))
        fig.add_trace(go.Scatter(x=future_dates, y=y_pred, mode="lines+markers", name="Forecast"))
        fig.update_layout(title=f"Forecast (next {N} periods)", template="plotly_dark")
        insights.append(f"Linear forecast model: coef={model.coef_[0]:.4f}, intercept={model.intercept_:.2f}")
        return [dcc.Graph(figure=fig)], total_sales, total_profit, aov, "\n".join(insights)

    # else: generate selected charts (multiple)
    charts = []
    # default chart_types if none selected
    if not chart_types:
        chart_types = ["line"]

    for ctype in chart_types:
        try:
            if ctype == "line" and date_col and sales_col:
                agg = dff.groupby(date_col).agg({sales_col: "sum"}).reset_index().sort_values(date_col)
                fig = px.line(agg, x=date_col, y=sales_col, title="Sales Over Time")
                fig.update_layout(template="plotly_dark")
                charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
            elif ctype == "bar":
                group_by = product_col or region_col or (det.get("categorical_cols") or [None])[0]
                if group_by and sales_col:
                    agg = dff.groupby(group_by).agg({sales_col: "sum"}).reset_index().sort_values(sales_col, ascending=False)
                    fig = px.bar(agg.head(30), x=group_by, y=sales_col, title=f"Top by {group_by}")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
                else:
                    # fallback histogram of sales
                    fig = px.histogram(dff, x=sales_col, title="Sales Distribution (fallback for bar)")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
            elif ctype == "hist":
                num = (dff.select_dtypes("number").columns.tolist() or [None])[0]
                if num:
                    fig = px.histogram(dff, x=num, nbins=30, title=f"Histogram - {num}")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
            elif ctype == "box":
                num = (dff.select_dtypes("number").columns.tolist() or [None])[0]
                if num:
                    fig = px.box(dff, y=num, title=f"Boxplot - {num}")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
            elif ctype == "pie":
                cat = product_col or region_col or (det.get("categorical_cols") or [None])[0]
                if cat:
                    vc = dff[cat].value_counts().reset_index()
                    vc.columns = [cat, "count"]
                    fig = px.pie(vc, names=cat, values="count", title=f"Pie - {cat}")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
            elif ctype == "scatter":
                nums = dff.select_dtypes("number").columns.tolist()
                if len(nums) >= 2:
                    fig = px.scatter(dff, x=nums[0], y=nums[1], title=f"Scatter: {nums[0]} vs {nums[1]}")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
            elif ctype == "corr":
                nums = dff.select_dtypes("number")
                if nums.shape[1] >= 2:
                    corr = nums.corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                    fig.update_layout(template="plotly_dark")
                    charts.append(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig)]), className="mb-3"))
        except Exception as e:
            charts.append(html.Div(f"Error creating {ctype} chart: {e}"))

    if not charts:
        charts = [html.Div("No charts could be generated with current selection.")]

    # quick insights
    try:
        insights.append(f"Rows after filter: {len(dff)}")
        if date_col:
            insights.append(f"Date range: {dff[date_col].min()} to {dff[date_col].max()}")
        if sales_col:
            insights.append(f"Total sales (filtered): {total_sales}")
        if profit_col:
            insights.append(f"Total profit (filtered): {total_profit}")
    except Exception:
        pass

    return charts, total_sales, total_profit, aov, "\n".join(insights)

# -------------------------
# Run server (Dash v3)
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
