import streamlit as st
import pandas as pd
from tools import default_rate_by, score_borrower, VALID, ALLOWED_COLUMNS

st.set_page_config(page_title="Credit Risk Copilot")
st.title("Credit Risk Copilot")
st.caption("Portfolio project on a public Kaggle dataset. Not a real lending decision tool.")

tab1, tab2, tab3 = st.tabs(["Explore", "Score a borrower", "Ask the copilot"])

with tab1:
    col = st.selectbox("Default rate by", ALLOWED_COLUMNS)
    df = pd.DataFrame(default_rate_by(col))
    st.bar_chart(df, x=col, y="default_rate")
    st.dataframe(df, hide_index=True)

with tab2:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 100, 30)
    income = c1.number_input("Annual income ($)", 4000, 500000, 60000, step=1000)
    emp = c1.number_input("Years employed", 0.0, 50.0, 4.0)
    loan = c1.number_input("Loan amount ($)", 500, 35000, 10000, step=500)
    rate = c2.number_input("Interest rate (%)", 5.0, 25.0, 11.0)
    hist = c2.number_input("Credit history (years)", 2, 40, 5)
    home = c2.selectbox("Home ownership", VALID['person_home_ownership'])
    intent = c2.selectbox("Loan intent", VALID['loan_intent'])
    grade = c2.selectbox("Loan grade", VALID['loan_grade'])
    prior = c2.selectbox("Past default on file", VALID['cb_person_default_on_file'])
    if st.button("Score"):
        p = score_borrower(age, income, emp, loan, rate, hist, home, intent, grade, prior)
        st.metric("Predicted default probability", f"{p:.1%}")

with tab3:
    from agent import run_agent
    st.session_state.setdefault("asked", 0)
    q = st.text_input("Ask a question about the data")
    if st.button("Ask") and q:
        if st.session_state.asked >= 5:
            st.warning("Question limit reached for this session.")
        else:
            st.session_state.asked += 1
            with st.spinner("Thinking..."):
                answer, log = run_agent(q)
            st.write(answer)
            with st.expander("Tools called"):
                st.write(log)