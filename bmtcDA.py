import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, trim_mean, pearsonr

# Page Setup
st.set_page_config(page_title="BMTC Statistical Analysis", layout="wide")
st.title("🚌 BMTC Financial Statistical Report")

# 1. File Upload
uploaded_file = st.file_uploader("Upload BMTC Financial CSV", type="csv")

if uploaded_file is not None:
    # --- Sample Data Preview ---
    st.header("1. Data Preview")
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Raw Sample from CSV")
    st.dataframe(df_raw.head())
    
    # --- Data Cleaning Logic ---
    target_factors = ["Through Sale of Tickets", "Monthly pass", "Daily pass", "Student pass", "Others", "Total"]
    df_raw['Factors_clean'] = df_raw['Factors'].str.strip().str.lower()
    
    selected = {}
    for t in target_factors:
        match = df_raw[df_raw['Factors_clean'].str.contains(t.lower().split()[0], na=False)]
        if len(match) > 0: selected[t] = match.iloc[0]
    
    df_sel = pd.DataFrame(selected).T
    year_cols = [c for c in df_sel.columns if ("20" in c or "19" in c) and "bifurcation" not in c]
    df = df_sel[year_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", ""), errors="coerce")).T
    df = df.dropna(how='all').fillna(0)

    st.subheader("Cleaned Data (Years as Rows, Factors as Columns)")
    st.dataframe(df)

    # --- Activity 2: Summary Statistics ---
    st.header("2. Summary Statistics")
    stats_dict = {
        "Mean": df.mean(),
        "Median": df.median(),
        "Trimmed Mean (10%)": df.apply(lambda x: trim_mean(x, 0.1)),
        "Std Dev": df.std(),
        "MAD": df.apply(lambda x: (x - x.mean()).abs().mean()),
        "IQR": df.quantile(0.75) - df.quantile(0.25)
    }
    st.dataframe(pd.DataFrame(stats_dict).T)

    # --- Activity 3: Time Trend, Box & Q-Q Plots ---
    st.header("3. Specific Factor Analysis (Trend & Normality)")
    selected_col = st.selectbox("Select a Factor to Analyze:", df.columns)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.subheader(f"Trend: {selected_col}")
        fig1, ax1 = plt.subplots()
        ax1.plot(df.index, df[selected_col], marker='o', linestyle='-', color='blue', linewidth=2)
        ax1.set_ylabel("Amount (Lakhs)")
        ax1.set_xlabel("Year")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.info(f"This line chart compares **Year (X)** and **{selected_col} (Y)**.")
    
    with col_b:
        st.subheader("Box Plot (Outliers)")
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=df[selected_col], ax=ax2, color="lightgreen")
        ax2.set_ylabel("Lakhs")
        st.pyplot(fig2)
        
    with col_c:
        st.subheader("Q-Q Plot (Normality)")
        fig3, ax3 = plt.subplots()
        stats.probplot(df[selected_col], dist="norm", plot=ax3)
        st.pyplot(fig3)

    # --- Activity 4: Correlation Matrix ---
    st.header("4. Correlation Matrix")
    corr = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # --- Activity 5: Hypothesis Testing ---
    st.header("5. Null Hypothesis Testing")
    st.markdown("""
    * **Null Hypothesis ($H_0$):** There is **no** significant linear relationship between the two selected factors.
    * **Alternative Hypothesis ($H_a$):** There **is** a significant linear relationship between them.
    """)
    
    col_x = st.selectbox("Independent Variable (X)", df.columns, index=0)
    col_y = st.selectbox("Dependent Variable (Y)", df.columns, index=5)
    
    if col_x != col_y:
        coeff, p_value = pearsonr(df[col_x], df[col_y])
        
        st.subheader("Statistical Test Results")
        st.write(f"**Correlation Coefficient (r):** {coeff:.4f}")
        st.write(f"**P-Value:** {p_value:.4e}")
        
        if p_value < 0.05:
            st.success(f"**Result:** Reject $H_0$. There is a strong, statistically significant relationship between {col_x} and {col_y}.")
        else:
            st.error(f"**Result:** Fail to Reject $H_0$. The relationship between {col_x} and {col_y} is not statistically significant.")
    else:
        st.warning("Please choose two different variables.")

    # --- Activity 6: Relationship Visuals ---
    st.header("6. Relationship Visualization")
    tab1, tab2 = st.tabs(["Scatter Plot (Regression)", "Violin Plot"])
    
    with tab1:
        fig4, ax4 = plt.subplots()
        sns.regplot(x=df[col_x], y=df[col_y], ax=ax4, color="purple")
        ax4.set_title(f"Relationship: {col_x} vs {col_y}")
        st.pyplot(fig4)
    with tab2:
        fig5, ax5 = plt.subplots()
        sns.violinplot(data=df[[col_x, col_y]], ax=ax5)
        st.pyplot(fig5)

    # --- Conclusion ---
    st.header("7. Final Conclusion for Presentation")
    st.write(f"""
    1.  **Time Trends:** The line chart for **{selected_col}** shows how the revenue changed across the years, clearly highlighting the impact of external events like the 2020 pandemic.
    2.  **Outliers:** The Box Plot shows that certain years (like the pandemic year) act as outliers, pulling the average down.
    3.  **Normality:** The Q-Q Plot indicates whether our growth follows a normal distribution.
    4.  **Hypothesis Summary:** Based on the test between **{col_x}** and **{col_y}**, we conclude that the relationship is **{'statistically significant' if p_value < 0.05 else 'not significant'}**.
    """)

else:
    st.info("Please upload your BMTC CSV file to start the live analysis.")