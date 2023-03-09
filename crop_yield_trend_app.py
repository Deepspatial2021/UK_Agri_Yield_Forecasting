

import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import RangeTool,HoverTool
from bokeh.layouts import column

st.set_page_config(page_title='Annual Crop Yield Prediction',layout="wide")

ets=pd.read_excel("Consolidated Predictions.xlsx",sheet_name="ETS").set_index("Date")
prophet=pd.read_excel("Consolidated Predictions.xlsx",sheet_name="Prophet").set_index("Date")
sarima=pd.read_excel("Consolidated Predictions.xlsx",sheet_name="SARIMA").set_index("Date")
auto_arima=pd.read_excel("Consolidated Predictions.xlsx",sheet_name="AUTO ARIMA").set_index("Date")
lr=pd.read_excel("Consolidated Predictions.xlsx",sheet_name="Linear Regression").set_index("Date")


#%%

st.markdown('<div style="text-align: center; font-size:30px; font-weight:bold">Crop Yield Prediction</div>', unsafe_allow_html=True)

st.markdown("<br></br>",unsafe_allow_html=True)

inf1,inf2,inf3=st.columns(3)

with inf1:
    st.subheader("Training Period : 1961-2007")

with inf2:
    st.subheader("Test Period : 2008-2019")

with inf3:
    st.subheader("Future Forecast Period : 2020-2024")

st.markdown("\n")
st.markdown("\n")

ovr_all,ets_t,sarima_t,prophet_t,auto_arima_t,lr_t=st.tabs(["Comparison","ETS",
                                                    "SARIMA","Prophet","AUTO ARIMA","Linear Regression"])


with ovr_all:
    all_fig = figure(title="Actual vs Predicted Crop Yield", x_axis_label="Date", 
       y_axis_label="Crop Yield (in Tonnes/ha)", x_axis_type="datetime",width=1500, height=400,tools='xpan',
     x_axis_location="above",x_range=(ets.index[10],ets.index[63]))
    all_fig.line(ets.index,ets['Actual'], legend_label="Actual", color="red",line_width=2,line_dash="dashdot")
    all_fig.line(ets.index,ets['Predicted'],legend_label="ETS Prediction",color="blue")
    all_fig.line(prophet.index,prophet['Predicted'],legend_label="Prophet Prediction",color="orange")
    all_fig.line(sarima.index,sarima['Predicted'],legend_label="SARIMA Prediction",color="green")
    all_fig.line(lr.index,lr['Predicted'],legend_label="Linear Regrression Prediction",color="violet")
    all_fig.line(auto_arima.index,auto_arima['Predicted'],legend_label="Auto ARIMA Prediction",color="grey")

    all_fig.legend.location='top_left' 
    all_fig.add_tools(HoverTool(tooltips=
        [
            ('Date',  '$data_x{%F}'),
            ('Crop Yield', '$data_y{0,0.00}'),
        ],
        formatters={
            '$data_x': 'datetime',
        }))
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=all_fig.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    
    range_tool = RangeTool(x_range=all_fig.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(ets.index,ets['Actual'],color="red")
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    st.bokeh_chart(column(all_fig, select), use_container_width=True)    

    st.markdown("\n")
    st.markdown("\n")
    
    models=["ETS","SARIMA","Prophet","AUTO ARIMA","Linear Regression"]
    train_acc=[100-int(ets.groupby("Label")['APE'].mean()['Train']),
               100-int(sarima.groupby("Label")['APE'].mean()['Train']),
               100-int(prophet.groupby("Label")['APE'].mean()['Train']),
               96,
               100-int(lr.groupby("Label")['APE'].mean()['Train'])]
    test_acc=[100-int(ets.groupby("Label")['APE'].mean()['Test']),
               100-int(sarima.groupby("Label")['APE'].mean()['Test']),
               100-int(prophet.groupby("Label")['APE'].mean()['Test']),
               100-int(auto_arima.groupby("Label")['APE'].mean()['Test']),
               100-int(lr.groupby("Label")['APE'].mean()['Test'])]
    
    compare_df=pd.DataFrame(zip(train_acc,test_acc),
                columns=['Train Accuracy (%)','Test Accuracy (%)'],index=models)

    col1,col2,col3=st.columns(3)
    with col2:
        st.write(compare_df)
    
with ets_t:
    q = figure(title="Actual vs Predicted Crop Yield", x_axis_label="Date", 
       y_axis_label="Crop Yield (in Tonnes/ha)", x_axis_type="datetime",width=1500, height=400,tools='xpan',
     x_axis_location="above",x_range=(ets.index[10],ets.index[63]))
    q.line(ets.index,ets['Actual'], legend_label="Actual", color="red",line_width=3)
    q.line(ets.index[:48],ets.loc[:"2008-04-01",'Predicted'],legend_label="Train Pred",color="blue")
    q.line(ets.index[47:60],ets.loc["2007-04-01":"2020-04-01",'Predicted'],legend_label="Test Pred",color="orange")
    q.line(ets.index[59:],ets.loc["2019-04-01":,'Predicted'],legend_label="Future Forecast",color="green")
    q.legend.location='top_left'
    q.add_tools(HoverTool(tooltips=
        [
            ('Date',  '$data_x{%F}'),
            ('Crop Yield', '$data_y{0,0.00}'),
        ],
        formatters={
            '$data_x': 'datetime',
        }))
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=q.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    
    range_tool = RangeTool(x_range=q.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(ets.index,ets['Actual'],color="red")
    select.line(ets.index,ets['Predicted'],color='blue')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    st.bokeh_chart(column(q, select), use_container_width=True)    
    st.markdown("\n")
    c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
    
    ets_test_mape=int(ets.groupby("Label")['APE'].mean()['Test'])
    ets_train_mape=int(ets.groupby("Label")['APE'].mean()['Train'])
    with c2:    
        st.metric("Test Accuracy",str(100-ets_test_mape)+"%")

    with c3:
        st.metric('Test MAPE',str(ets_test_mape)+"%")
        
    with c6:    
        st.metric("Train Accuracy",str(100-ets_train_mape)+"%")

    with c7:
        st.metric('Train MAPE',str(ets_train_mape)+"%")
    
    
    
with sarima_t:
    w = figure(title="Actual vs Predicted Crop Yield", x_axis_label="Date", 
       y_axis_label="Crop Yield (in Tonnes/ha)", x_axis_type="datetime",width=1500, height=400,tools='xpan',
     x_axis_location="above",x_range=(sarima.index[10],sarima.index[62]))
    w.line(sarima.index,sarima['Actual'], legend_label="Actual", color="red",line_width=2)
    w.line(sarima.index[:47],sarima.loc[:"2008-04-01",'Predicted'],legend_label="Train Pred",color="blue")
    w.line(sarima.index[46:59],sarima.loc["2007-04-01":"2020-04-01",'Predicted'],legend_label="Test Pred",color="orange")
    w.line(sarima.index[58:],sarima.loc["2019-04-01":,'Predicted'],legend_label="Future Forecast",color="green")
    w.legend.location='top_left'
    w.add_tools(HoverTool(tooltips=
        [
            ('Date',  '$data_x{%F}'),
            ('Crop Yield', '$data_y{0,0.00}'),
        ],
        formatters={
            '$data_x': 'datetime',
        }))
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=w.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    
    range_tool = RangeTool(x_range=w.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(sarima.index,sarima['Actual'],color="red")
    select.line(sarima.index,sarima['Predicted'],color='blue')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    st.bokeh_chart(column(w, select), use_container_width=True)    

    st.markdown("\n")
    c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
    
    sarima_test_mape=int(sarima.groupby("Label")['APE'].mean()['Test'])
    sarima_train_mape=int(sarima.groupby("Label")['APE'].mean()['Train'])
    with c2:    
        st.metric("Test Accuracy",str(100-sarima_test_mape)+"%")

    with c3:
        st.metric('Test MAPE',str(sarima_test_mape)+"%")
        
    with c6:    
        st.metric("Train Accuracy",str(100-sarima_train_mape)+"%")

    with c7:
        st.metric('Train MAPE',str(sarima_train_mape)+"%")
        
        
with prophet_t:
    r = figure(title="Actual vs Predicted Crop Yield", x_axis_label="Date", 
       y_axis_label="Crop Yield (in Tonnes/ha)", x_axis_type="datetime",width=1500, height=400,tools='xpan',
     x_axis_location="above",x_range=(prophet.index[10],prophet.index[63]))
    r.line(prophet.index,prophet['Actual'], legend_label="Actual", color="red",line_width=2)
    r.line(prophet.index[:48],prophet.loc[:"2008-04-01",'Predicted'],legend_label="Train Pred",color="blue")
    r.line(prophet.index[47:60],prophet.loc["2007-04-01":"2020-04-01",'Predicted'],legend_label="Test Pred",color="orange")
    r.line(prophet.index[59:],prophet.loc["2019-04-01":,'Predicted'],legend_label="Future Forecast",color="green")
    r.legend.location='top_left'
    r.add_tools(HoverTool(tooltips=
        [
            ('Date',  '$data_x{%F}'),
            ('Crop Yield', '$data_y{0,0.00}'),
        ],
        formatters={
            '$data_x': 'datetime',
        }))
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=r.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    
    range_tool = RangeTool(x_range=r.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(prophet.index,prophet['Actual'],color="red")
    select.line(prophet.index,prophet['Predicted'],color='blue')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    st.bokeh_chart(column(r, select), use_container_width=True)    

    st.markdown("\n")
    c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
    
    prophet_test_mape=int(prophet.groupby("Label")['APE'].mean()['Test'])
    prophet_train_mape=int(prophet.groupby("Label")['APE'].mean()['Train'])
    with c2:    
        st.metric("Test Accuracy",str(100-prophet_test_mape)+"%")

    with c3:
        st.metric('Test MAPE',str(prophet_test_mape)+"%")
        
    with c6:    
        st.metric("Train Accuracy",str(100-prophet_train_mape)+"%")

    with c7:
        st.metric('Train MAPE',str(prophet_train_mape)+"%")


with auto_arima_t:
    zx = figure(title="Actual vs Predicted Crop Yield", x_axis_label="Date", 
       y_axis_label="Crop Yield (in Tonnes/ha)", x_axis_type="datetime",width=1500, height=400,tools='xpan',
     x_axis_location="above",x_range=(auto_arima.index[3],auto_arima.index[16]))
    zx.line(auto_arima.index,auto_arima['Actual'], legend_label="Actual", color="red",line_width=2)
    zx.line(auto_arima.index[:13],auto_arima.loc["2007-04-01":"2020-04-01",'Predicted'],legend_label="Test Pred",color="orange")
    zx.line(auto_arima.index[12:],auto_arima.loc["2019-04-01":,'Predicted'],legend_label="Future Forecast",color="green")
    zx.legend.location='top_left'
    zx.add_tools(HoverTool(tooltips=
        [
            ('Date',  '$data_x{%F}'),
            ('Crop Yield', '$data_y{0,0.00}'),
        ],
        formatters={
            '$data_x': 'datetime',
        }))
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=zx.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    
    range_tool = RangeTool(x_range=zx.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(auto_arima.index,auto_arima['Actual'],color="red")
    select.line(auto_arima.index,auto_arima['Predicted'],color='blue')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    st.bokeh_chart(column(zx, select), use_container_width=True)    

    st.markdown("\n")
    c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
    
    auto_arima_test_mape=int(auto_arima.groupby("Label")['APE'].mean()['Test'])
    with c3:    
        st.metric("Test Accuracy",str(100-auto_arima_test_mape)+"%")

    with c6:
        st.metric('Test MAPE',str(auto_arima_test_mape)+"%")
        


with lr_t:
    xz = figure(title="Actual vs Predicted Crop Yield", x_axis_label="Date", 
       y_axis_label="Crop Yield (in Tonnes/ha)", x_axis_type="datetime",width=1500, height=400,tools='xpan',
     x_axis_location="above",x_range=(lr.index[8],lr.index[54]))
    xz.line(lr.index,lr['Actual'], legend_label="Actual", color="red",line_width=2)
    xz.line(lr.index[:42],lr.loc[:"2008-04-01",'Predicted'],legend_label="Train Pred",color="blue")
    xz.line(lr.index[41:54],lr.loc["2007-04-01":"2020-04-01",'Predicted'],legend_label="Test Pred",color="orange")
    xz.line(lr.index[53:],lr.loc["2019-04-01":,'Predicted'],legend_label="Future Forecast",color="green")
    xz.legend.location='top_left'
    xz.add_tools(HoverTool(tooltips=
        [
            ('Date',  '$data_x{%F}'),
            ('Crop Yield', '$data_y{0,0.00}'),
        ],
        formatters={
            '$data_x': 'datetime',
        }))
    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=1400, y_range=xz.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")
    
    range_tool = RangeTool(x_range=xz.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.line(lr.index,lr['Actual'],color="red")
    select.line(lr.index,lr['Predicted'],color='blue')
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    st.bokeh_chart(column(xz, select), use_container_width=True)    
    
    st.markdown("\n")
    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10=st.columns(10)
    
    lr_test_mape=int(lr.groupby("Label")['APE'].mean()['Test'])
    lr_train_mape=int(lr.groupby("Label")['APE'].mean()['Train'])
    with c3:    
        st.metric("Test Accuracy",str(100-lr_test_mape)+"%")

    with c4:
        st.metric('Test MAPE',str(lr_test_mape)+"%")

    with c6:
        st.metric("Crossvalidated MAPE","6%")        
        
    with c8:    
        st.metric("Train Accuracy",str(100-lr_train_mape)+"%")

    with c9:
        st.metric('Train MAPE',str(lr_train_mape)+"%")


