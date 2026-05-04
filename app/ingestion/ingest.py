import pandas as pd

def create_prompt(filepath : str):
    st=pd.read_excel("../tele.xlsx")
    return st.prompt.tolist()

