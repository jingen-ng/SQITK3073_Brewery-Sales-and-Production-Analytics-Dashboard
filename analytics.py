import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BreweryAnalytics:


    def __init__(self, df=None, db_path="brewery_sales.db"):


        self.models = {}

        # =====================================
        # IF STREAMLIT SENDS CSV DATA
        # =====================================

        if df is not None:


            self.data = self._engineer_production_data(
                df
            )


            # optional sqlite backup
            self.conn = sqlite3.connect(
                db_path,
                check_same_thread=False
            )


            self._save_to_sql(
                self.data,
                "brewery_data"
            )


        else:


            # keep old database support

            self.conn = sqlite3.connect(
                db_path,
                check_same_thread=False
            )


            self.data = None

    # ---------------------- dataload and clear ----------------------
    def _deduplicate_columns(self, df):
        if not df.columns.has_duplicates:
            return df

        cleaned = pd.DataFrame(index=df.index)
        for col in dict.fromkeys(df.columns):
            same_name_cols = df.loc[:, df.columns == col]
            if isinstance(same_name_cols, pd.Series) or same_name_cols.shape[1] == 1:
                cleaned[col] = same_name_cols.squeeze()
            else:
                cleaned[col] = same_name_cols.bfill(axis=1).iloc[:, 0]
        return cleaned

    def load_txt_data(self, row_limit=None, local_file="brewery_50k_all_dates.csv"):
        try:
            df = pd.read_csv(local_file)
            if row_limit is not None and len(df) > row_limit:
                df = df.head(row_limit)
            self.data = self._engineer_production_data(df)
            self._save_to_sql(self.data, "brewery_data")
            return f"Loaded {len(self.data)} records from {local_file}"
        except Exception as e:
            return f"Error: {str(e)}"

    def load_huggingface_data(self, sample_size=10000000):
        try:
            from datasets import load_dataset
            dataset = load_dataset("Mathi65xl/Brewery_sales", split=f"train[:{sample_size}]")
            df = pd.DataFrame(dataset)
            self.data = self._engineer_production_data(df)
            self._save_to_sql(self.data, "brewery_data")
            return f"Loaded {len(self.data)} records with production metrics"
        except Exception as e:
            return f"Error: {str(e)}"

    def upload_custom_data(self, file):
        try:
            df = pd.read_csv(file)
            self.data = self._engineer_production_data(df)
            self._save_to_sql(self.data, "brewery_data")
            return f"Imported {len(self.data)} records"
        except Exception as e:
            return f"Error: {str(e)}"

    def _engineer_production_data(self, df):

        df = df.copy()

        df.columns = df.columns.str.lower().str.strip()
        df = self._deduplicate_columns(df)

  
        col_map = {
            "brand": "beer_style", "beer_name": "beer_style", "product_brand": "beer_style",
            "city": "location", "store": "location",
            "sku": "sku",
            "quantity": "production_volume", "units_sold": "production_volume", "volume_produced": "production_volume",
            "price": "unit_price", "unit_price": "unit_price",
            "sales": "total_revenue", "total_sales": "total_revenue", "revenue": "total_revenue",
            "date": "sale_date", "brew_date": "sale_date", "start_of_week": "sale_date", "week": "week_number",
            "fermentation_time": "fermentation_days", "quality": "quality_score",
            "brewhouse_efficiency": "production_efficiency",
            "loss_during_fermentation": "production_loss",
            "loss_during_brewing": "brewing_loss",
            "loss_during_bottling_kegging": "packaging_loss",
            "bitterness": "bitterness_ibu"
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = self._deduplicate_columns(df)


        if "beer_style" not in df.columns:
            df["beer_style"] = np.random.choice(["Lager", "IPA", "Stout", "Wheat", "Pale Ale"], len(df))
        if "location" not in df.columns:
            df["location"] = np.random.choice(["Kuala Lumpur", "Penang", "Johor", "Sabah", "Sarawak"], len(df))
        if "sku" not in df.columns:
            df["sku"] = "Unknown"
        if "production_volume" not in df.columns:
            df["production_volume"] = np.random.randint(100, 5000, len(df))
        df["production_volume"] = pd.to_numeric(df["production_volume"], errors="coerce").fillna(0)
        if "unit_price" not in df.columns:
            if "total_revenue" in df.columns:
                df["unit_price"] = df["total_revenue"] / df["production_volume"].replace(0, np.nan)
                df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
            else:
                df["unit_price"] = np.random.uniform(5, 25, len(df))
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)

        # date format
        if "sale_date" not in df.columns:
            df["sale_date"] = pd.date_range("2022-01-01", periods=len(df), freq="D")
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
        df["year"] = df["sale_date"].dt.year
        df["month"] = df["sale_date"].dt.month

        # cal revenue
        if "total_revenue" not in df.columns:
            df["total_revenue"] = df["production_volume"] * df["unit_price"]
        df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce").fillna(0)

        np.random.seed(42)
        base_ferm = {"Lager": 14, "IPA": 10, "Stout": 18, "Wheat": 7, "Pale Ale": 12}
        if "fermentation_days" not in df.columns:
            mapped_ferm = df["beer_style"].map(base_ferm).fillna(12)
            df["fermentation_days"] = mapped_ferm + np.random.normal(0, 2, len(df)).round(1)
        df["fermentation_days"] = pd.to_numeric(df["fermentation_days"], errors="coerce")
        df["fermentation_days"] = df["fermentation_days"].fillna(df["fermentation_days"].median())
        df["fermentation_days"] = df["fermentation_days"].clip(3, 30)  # 异常值裁剪

        base_quality = {"Lager": 82, "IPA": 88, "Stout": 90, "Wheat": 78, "Pale Ale": 85}
        if "quality_score" not in df.columns:
            mapped_quality = df["beer_style"].map(base_quality).fillna(85)
            df["quality_score"] = mapped_quality + np.random.normal(0, 4, len(df)).round(1)
        df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
        if df["quality_score"].max() <= 10:
            df["quality_score"] = df["quality_score"] * 10
        df["quality_score"] = df["quality_score"].fillna(df["quality_score"].median())
        df["quality_score"] = df["quality_score"].clip(60, 100)

        base_abv = {"Lager": 4.5, "IPA": 6.8, "Stout": 7.2, "Wheat": 5.0, "Pale Ale": 5.5}
        if "alcohol_content" not in df.columns:
            mapped_abv = df["beer_style"].map(base_abv).fillna(5.5)
            df["alcohol_content"] = mapped_abv + np.random.normal(0, 0.5, len(df)).round(2)
        df["alcohol_content"] = pd.to_numeric(df["alcohol_content"], errors="coerce").fillna(df["alcohol_content"].median())

        base_ibu = {"Lager": 18, "IPA": 65, "Stout": 45, "Wheat": 15, "Pale Ale": 35}
        if "bitterness_ibu" not in df.columns:
            mapped_ibu = df["beer_style"].map(base_ibu).fillna(35)
            df["bitterness_ibu"] = mapped_ibu + np.random.normal(0, 8, len(df)).round(1)
        df["bitterness_ibu"] = pd.to_numeric(df["bitterness_ibu"], errors="coerce").fillna(df["bitterness_ibu"].median())

        if "production_loss" not in df.columns:
            df["production_loss"] = (df["production_volume"] * np.random.uniform(0.02, 0.12, len(df))).round(0)
        df["production_loss"] = pd.to_numeric(df["production_loss"], errors="coerce").fillna(0)
        for loss_col in ["brewing_loss", "packaging_loss"]:
            if loss_col in df.columns:
                df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce").fillna(0)
        if "production_efficiency" not in df.columns:
            df["production_efficiency"] = ((df["production_volume"] - df["production_loss"]) / df["production_volume"].replace(0, np.nan) * 100).round(2)
        df["production_efficiency"] = pd.to_numeric(df["production_efficiency"], errors="coerce").fillna(df["production_efficiency"].median())
        df["revenue_per_ferm_day"] = (df["total_revenue"] / df["fermentation_days"]).round(2)

        return df

    def _save_to_sql(self, df, table_name, if_exists="replace"):
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)

    def run_sql(self, query):
        return pd.read_sql(query, self.conn)

    def _filter_by_year(self, df, year=None):
        df = self._deduplicate_columns(df)
        if year is None or year == []:
            return df
        if isinstance(year, (list, tuple, set)):
            return df[df["year"].isin(year)]
        return df[df["year"] == year]

    # ---------------------- analytics ----------------------
    def get_kpis(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)

        total_revenue = df["total_revenue"].sum()
        total_production = df["production_volume"].sum()
        best_style = df.groupby("beer_style")["total_revenue"].sum().idxmax()
        best_quality = df.groupby("beer_style")["quality_score"].mean().idxmax()
        avg_efficiency = df["production_efficiency"].mean()

        return {
            "total_revenue": total_revenue,
            "total_production": total_production,
            "best_selling_style": best_style,
            "highest_quality_style": best_quality,
            "avg_efficiency": avg_efficiency
        }

    def get_production_by_month(self, year=None):
        df = self._deduplicate_columns(self.data.copy())
        if year:
            df = df[df["year"].isin(year)] if isinstance(year, (list, tuple, set)) else df[df["year"] == year]
        return df.groupby("month")["production_volume"].sum().reset_index()

    def get_production_by_year(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        return df.groupby("year")["production_volume"].sum().reset_index().sort_values("year")

    def get_efficiency_by_style(self, year=None):
        df = self._deduplicate_columns(self.data.copy())
        if year:
            df = df[df["year"].isin(year)] if isinstance(year, (list, tuple, set)) else df[df["year"] == year]
        return df.groupby("beer_style").agg(
            avg_efficiency=("production_efficiency", "mean"),
            avg_ferm_days=("fermentation_days", "mean")
        ).reset_index().round(2)

    def get_product_performance(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        group_cols = ["beer_style", "sku"] if "sku" in df.columns else ["beer_style"]
        return df.groupby(group_cols).agg(
            total_production=("production_volume", "sum"),
            total_revenue=("total_revenue", "sum"),
            avg_quality=("quality_score", "mean"),
            avg_efficiency=("production_efficiency", "mean")
        ).reset_index().round(2).sort_values("total_revenue", ascending=False)

    def get_ferm_efficiency_correlation(self):
        return round(self.data["fermentation_days"].corr(self.data["production_efficiency"]), 4)

    def get_revenue_per_ferm_day(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        return df.groupby("beer_style")["revenue_per_ferm_day"].mean().reset_index().round(2).sort_values("revenue_per_ferm_day", ascending=False)

    def get_best_style_by_location(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        return df.groupby(["location", "beer_style"])["total_revenue"].sum().reset_index() \
            .sort_values("total_revenue", ascending=False) \
            .drop_duplicates("location").sort_values("location")

    def get_best_quality_by_year(self):
        df = self._deduplicate_columns(self.data.copy())
        return df.groupby(["year", "beer_style"])["quality_score"].mean().reset_index().round(2)

    # visualization data
    def get_sales_by_style(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        return df.groupby("beer_style")["total_revenue"].sum().reset_index().sort_values("total_revenue", ascending=False)

    def get_sales_vs_production(self, year=None):
        df = self._deduplicate_columns(self.data.copy())
        if year:
            df = df[df["year"].isin(year)] if isinstance(year, (list, tuple, set)) else df[df["year"] == year]
        return df.groupby("beer_style").agg(
            revenue=("total_revenue", "sum"),
            volume=("production_volume", "sum")
        ).reset_index()

    def get_sku_sales_vs_production(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        if "sku" not in df.columns:
            df["sku"] = df["beer_style"]
        return df.groupby("sku").agg(
            revenue=("total_revenue", "sum"),
            volume=("production_volume", "sum"),
            avg_quality=("quality_score", "mean"),
            avg_efficiency=("production_efficiency", "mean")
        ).reset_index().round(2).sort_values("revenue", ascending=False)

    def get_loss_by_style(self, year=None):
        df = self._deduplicate_columns(self.data.copy())
        if year:
            df = df[df["year"].isin(year)] if isinstance(year, (list, tuple, set)) else df[df["year"] == year]
        return df.groupby("beer_style")["production_loss"].sum().reset_index()

    def get_total_loss_by_sku(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        if "sku" not in df.columns:
            df["sku"] = df["beer_style"]
        return (
            df.groupby("sku")["production_loss"]
            .sum()
            .reset_index()
            .round(2)
            .sort_values("production_loss", ascending=False)
        )

    def get_loss_breakdown_by_style(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)
        if "brewing_loss" not in df.columns:
            df["brewing_loss"] = df["production_loss"] * 0.35
        if "packaging_loss" not in df.columns:
            df["packaging_loss"] = df["production_loss"] * 0.25

        loss_data = df.groupby("beer_style")[["brewing_loss", "production_loss", "packaging_loss"]].sum()
        loss_data = loss_data.rename(columns={
            "brewing_loss": "Brewing Loss",
            "production_loss": "Fermentation Loss",
            "packaging_loss": "Packaging Loss"
        })
        return loss_data.reset_index().melt(
            id_vars="beer_style",
            var_name="loss_stage",
            value_name="loss_amount"
        ).round(2)

    def get_quality_trend(self):
        df = self._deduplicate_columns(self.data.copy())
        return df.groupby(["year", "beer_style"])["quality_score"].mean().reset_index()

    def get_alcohol_trend(self):
        df = self._deduplicate_columns(self.data.copy())
        return df.groupby(["year", "beer_style"])["alcohol_content"].mean().reset_index()

    def get_bitterness_trend(self):
        df = self._deduplicate_columns(self.data.copy())
        return df.groupby(["year", "beer_style"])["bitterness_ibu"].mean().reset_index()

    def get_ferm_vs_quality(self):
        return self.data[["fermentation_days", "quality_score", "beer_style"]]

    # business insights
    def get_descriptive_insights(self, year=None):
        df = self._filter_by_year(self.data.copy(), year)

        style_revenue = df.groupby("beer_style")["total_revenue"].sum().sort_values(ascending=False)
        style_quality = df.groupby("beer_style")["quality_score"].mean().sort_values(ascending=False)
        sku_col = "sku" if "sku" in df.columns else "beer_style"
        sku_summary = df.groupby(sku_col).agg(
            total_revenue=("total_revenue", "sum"),
            total_production=("production_volume", "sum"),
            total_loss=("production_loss", "sum")
        ).reset_index()
        sku_summary["revenue_per_unit"] = sku_summary["total_revenue"] / sku_summary["total_production"].replace(0, np.nan)
        sku_summary["loss_rate"] = sku_summary["total_loss"] / sku_summary["total_production"].replace(0, np.nan)
        sku_summary = sku_summary.fillna(0)
        sku_summary = sku_summary.sort_values("revenue_per_unit", ascending=False)

        highest_rev_style = style_revenue.index[0]
        highest_quality_style = style_quality.index[0]
        lowest_rev_style = style_revenue.index[-1]
        best_performance = sku_summary.iloc[0]
        improvement_sku = sku_summary.iloc[-1]

        return {
            "highest_revenue_style": highest_rev_style,
            "highest_quality_style": highest_quality_style,
            "best_performance_sku": best_performance[sku_col],
            "improvement_sku": improvement_sku[sku_col],
            "style_performance_insight": (
                f"{highest_rev_style} leads beer-style revenue with RM {style_revenue.iloc[0]:,.2f}, "
                f"while {lowest_rev_style} contributes the least at RM {style_revenue.iloc[-1]:,.2f}. "
                f"Use the leader as the benchmark for assortment, pricing, and promotion decisions."
            ),
            "sku_performance_insight": (
                f"{best_performance[sku_col]} is the best performance SKU because it has the highest total revenue per production volume at RM {best_performance['revenue_per_unit']:,.2f}. "
                f"{improvement_sku[sku_col]} needs improvement because it has the lowest total revenue per production volume at RM {improvement_sku['revenue_per_unit']:,.2f}; review pricing, demand, production allocation, and loss control."
            )
        }

    # ---------------------- strategic ----------------------
    def get_strategic_data(self, year=None, location=None):
        df = self._filter_by_year(self.data.copy(), year)
        if location:
            df = df[df["location"] == location]

        rev_per_day_style = df.groupby("beer_style")["revenue_per_ferm_day"].mean().reset_index().sort_values("revenue_per_ferm_day", ascending=False)
        rev_per_day_loc = df.groupby("location")["revenue_per_ferm_day"].mean().reset_index().sort_values("revenue_per_ferm_day", ascending=False)
        total_rev_style = df.groupby("beer_style")["total_revenue"].sum().reset_index()

        all_data = self._deduplicate_columns(self.data.copy())
        yearly_rev = all_data.groupby(["year", "beer_style"])["total_revenue"].sum().reset_index()
        if len(yearly_rev["year"].unique()) >= 2:
            growth = yearly_rev.pivot(index="beer_style", columns="year", values="total_revenue")
            years = sorted(growth.columns)
            growth["growth_rate_pct"] = ((growth[years[-1]] - growth[years[-2]]) / growth[years[-2]] * 100).round(2)
            growth = growth.reset_index()[["beer_style", "growth_rate_pct"]]
        else:
            growth = pd.DataFrame({"beer_style": total_rev_style["beer_style"], "growth_rate_pct": np.random.uniform(5, 25, len(total_rev_style)).round(2)})

        total_rev_loc = df.groupby("location")["total_revenue"].sum().reset_index().sort_values("total_revenue", ascending=False)

        best_short_style = rev_per_day_style.iloc[0]["beer_style"]
        best_loc_per_style = df.groupby(["beer_style", "location"])["total_revenue"].sum().reset_index() \
            .sort_values("total_revenue", ascending=False).drop_duplicates("beer_style")

        best_long_style = total_rev_style.merge(growth, on="beer_style").sort_values("total_revenue", ascending=False).iloc[0]["beer_style"]
        best_expansion_loc = total_rev_loc.iloc[-1]["location"]

        return {
            "rev_per_day_style": rev_per_day_style,
            "rev_per_day_location": rev_per_day_loc,
            "total_rev_style": total_rev_style,
            "growth_rate": growth,
            "total_rev_location": total_rev_loc,
            "best_short_style": best_short_style,
            "best_location_per_style": best_loc_per_style,
            "best_long_style": best_long_style,
            "best_expansion_location": best_expansion_loc
        }

    # ----------------------forecast ----------------------
    def holt_exponential_forecast(self, values, alpha=0.6, beta=0.3, periods=3):
        values = list(values)
        if len(values) == 0:
            return []
        if len(values) == 1:
            return [values[0]] * periods

        level = values[0]
        trend = values[1] - values[0]

        for value in values[1:]:
            previous_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1 - beta) * trend

        return [level + i * trend for i in range(1, periods + 1)]

    def forecast_total_revenue_by_style(self, forecast_years=None):
        if forecast_years is None:
            forecast_years = [2024, 2025, 2026, 2027, 2028]

        source = self._deduplicate_columns(self.data.copy())
        df = source[source["year"] <= 2023].copy()
        all_forecasts = []

        for style in sorted(df["beer_style"].dropna().unique()):
            style_df = df[df["beer_style"] == style]
            yearly_data = (
                style_df.groupby("year")["total_revenue"]
                .sum()
                .reset_index()
                .rename(columns={"total_revenue": "value"})
                .sort_values("year")
            )

            if yearly_data.empty:
                continue

            historical = yearly_data.copy()
            historical["type"] = "Historical"
            historical["beer_style"] = style

            future_values = self.holt_exponential_forecast(
                yearly_data["value"],
                alpha=0.6,
                beta=0.3,
                periods=len(forecast_years)
            )
            forecast = pd.DataFrame({
                "year": forecast_years,
                "value": future_values,
                "type": "Forecast",
                "beer_style": style
            })

            all_forecasts.append(pd.concat([historical, forecast], ignore_index=True))

        if not all_forecasts:
            return pd.DataFrame(columns=["year", "value", "type", "beer_style"])

        result = pd.concat(all_forecasts, ignore_index=True)
        result["value"] = result["value"].clip(lower=0).round(2)
        return result

    def get_total_revenue_forecast_summary(self, forecast_data):
        forecast_only = forecast_data[forecast_data["type"] == "Forecast"].copy()
        if forecast_only.empty:
            return pd.DataFrame()

        return forecast_only.pivot_table(
            index="year",
            columns="beer_style",
            values="value",
            aggfunc="sum"
        ).reset_index().round(2)

    def train_forecast_models(self, beer_style=None):
        df = self._deduplicate_columns(self.data.copy())
        if beer_style:
            df = df[df["beer_style"] == beer_style]

        monthly = df.groupby(["year", "month"]).agg(
            revenue=("total_revenue", "sum"),
            production=("production_volume", "sum"),
            loss=("production_loss", "sum")
        ).reset_index()

        if len(monthly) < 6:
            return None, "Not enough data for forecasting"

        monthly["time_index"] = np.arange(len(monthly))
        X = monthly[["time_index"]]

        targets = {"revenue": "revenue", "production": "production", "loss": "loss"}
        metrics = {}

        for name, target in targets.items():
            y = monthly[target]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.models[name] = model
            pred = model.predict(X)
            metrics[name] = {
                "mae": round(mean_absolute_error(y, pred), 2),
                "r2": round(r2_score(y, pred), 4)
            }

        return monthly, metrics

    def generate_forecast(self, years_ahead=3):
        if not self.models:
            return None

        source = self._deduplicate_columns(self.data.copy())
        last_idx = len(source.groupby(["year", "month"]).size())
        future_steps = years_ahead * 12
        future_X = pd.DataFrame({"time_index": np.arange(last_idx, last_idx + future_steps)})

        last_year = source["year"].max()
        future_dates = pd.date_range(start=f"{last_year+1}-01-01", periods=future_steps, freq="MS")

        forecast = pd.DataFrame({"date": future_dates})
        forecast["year"] = forecast["date"].dt.year
        forecast["month"] = forecast["date"].dt.month

        for name, model in self.models.items():
            forecast[f"predicted_{name}"] = model.predict(future_X).round(2)

        yearly_forecast = forecast.groupby("year").agg(
            predicted_revenue=("predicted_revenue", "sum"),
            predicted_production=("predicted_production", "sum"),
            predicted_loss=("predicted_loss", "sum")
        ).reset_index()

        return forecast, yearly_forecast

    def get_predictive_insights(self, yearly_forecast):
        if yearly_forecast is None:
            return {}

        top_rev_year = yearly_forecast.loc[yearly_forecast["predicted_revenue"].idxmax(), "year"]
        highest_loss_year = yearly_forecast.loc[yearly_forecast["predicted_loss"].idxmax(), "year"]
        growth = ((yearly_forecast.iloc[-1]["predicted_revenue"] - yearly_forecast.iloc[0]["predicted_revenue"])
                  / yearly_forecast.iloc[0]["predicted_revenue"] * 100).round(1)

        return {
            "top_revenue_year": int(top_rev_year),
            "highest_loss_risk_year": int(highest_loss_year),
            "projected_growth_pct": growth,
            "increase_production_rec": f"Gradually increase production capacity by {growth}% over the forecast period to match projected revenue growth.",
            "loss_risk_rec": f"Proactively invest in quality control and process optimization before {highest_loss_year} to mitigate rising production loss risk."
        }
