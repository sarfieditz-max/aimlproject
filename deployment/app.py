import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import streamlit as st
from predict import predict_segment, segment_table

st.set_page_config(page_title='PurchaseSense - Shopper Segments', page_icon='🛒')
st.title('🛒 PurchaseSense - Shopper Segment Finder')
st.write('Enter what a visitor did in one session. The K-Means model puts them in the **nearest of 3 shopper segments**. '
         'Most ordinary sessions land in *Casual browsers* (74% of training sessions), so use the example buttons to see the others.')

# ---- example sessions (typical values of each segment) ----
PRESETS = {
    'Quick exit':       dict(admin=0, admin_d=0.0,   info=0, info_d=0.0,   prod=2,  prod_d=30.0,   bounce=0.18,  exit_=0.19, special=0.0, page_val=0.0),
    'Casual browser':   dict(admin=2, admin_d=60.0,  info=0, info_d=0.0,   prod=20, prod_d=700.0,  bounce=0.01,  exit_=0.035, special=0.0, page_val=0.0),
    'Engaged shopper':  dict(admin=6, admin_d=250.0, info=2, info_d=120.0, prod=85, prod_d=3500.0, bounce=0.006, exit_=0.02, special=0.0, page_val=0.0)}
for k, v in PRESETS['Casual browser'].items():
    st.session_state.setdefault(k, v)

def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

b1, b2, b3 = st.columns(3)
for col, name in zip((b1, b2, b3), PRESETS):
    col.button('Load: ' + name, on_click=load_preset, args=(name,))

c1, c2 = st.columns(2)
with c1:
    admin   = st.number_input('Administrative pages viewed', 0, 50, key='admin')
    admin_d = st.number_input('Time on administrative pages (seconds)', 0.0, 4000.0, key='admin_d')
    info    = st.number_input('Informational pages viewed', 0, 30, key='info')
    info_d  = st.number_input('Time on informational pages (seconds)', 0.0, 3000.0, key='info_d')
    prod    = st.number_input('Product pages viewed', 0, 800, key='prod')
with c2:
    prod_d   = st.number_input('Time on product pages (seconds)', 0.0, 70000.0, key='prod_d')
    bounce   = st.number_input('Bounce rate', 0.0, 0.2, step=0.005, format='%.3f', key='bounce')
    exit_    = st.number_input('Exit rate', 0.0, 0.2, step=0.005, format='%.3f', key='exit_')
    special  = st.selectbox('Closeness to a special day (0 = far, 1 = the day)', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], key='special')
    page_val = st.number_input('Page value (not used by this model)', 0.0, 400.0, key='page_val')

# live result: updates every time an input changes
res = predict_segment({'Administrative': admin, 'Administrative_Duration': admin_d, 'Informational': info,
                       'Informational_Duration': info_d, 'ProductRelated': prod, 'ProductRelated_Duration': prod_d,
                       'BounceRates': bounce, 'ExitRates': exit_, 'SpecialDay': special, 'PageValues': page_val})
st.divider()
st.subheader(f"Segment: {res['segment']}")
st.metric('Purchase rate of this segment (training data)', f"{res['segment_training_purchase_rate']*100:.1f}%",
          delta=f"{(res['segment_training_purchase_rate']-res['overall_purchase_rate'])*100:+.1f} pts vs overall")
st.info(res['advice'])

st.write('**How close is this session to each segment?** (a high share for two segments = a borderline session)')
st.bar_chart(pd.Series(res['closeness'], name='closeness').sort_values())

with st.expander('What does a typical session of each segment look like?'):
    st.dataframe(segment_table().round(3))
st.caption('Segments come from unsupervised K-Means; this is a segmentation tool, not a purchase predictor. '
           'Page value is ignored because the final model (V09) does not use it.')
