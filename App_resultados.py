import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title='TFT Predicciones', page_icon="⚡", layout='wide', initial_sidebar_state='auto')

st.markdown(
        f""" <style>.reportview-container .main .block-container{{
        max-width: {1450}px;
        padding-top: {0}rem;
        padding-right: {10}rem;
        padding-left: {10}rem;
        padding-bottom: {0}rem;
    }}
</style> 
""", unsafe_allow_html=True
    )
st.markdown("""
    <div style="background-color:#FF815F;padding:0px">
    <h1 style="color:#FFFFFF ;text-align:center;">Predicción del precio de la electricidad</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>A continuación podrá visualizar los resultados obtenidos "
            "con el modelo de Transformers de Fusión Temporal</h4>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 230px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 175px;
        margin-left: -400px;
    }
    """,
    unsafe_allow_html=True,
)

datos = pd.read_excel("precios.xlsx")
test_metrics = pd.read_excel("Test_metrics.xlsx")
predicciones = pd.read_excel("Predicciones.xlsx")
attention = pd.read_excel("Atención.xlsx")


@st.cache(suppress_st_warning=True, show_spinner=False)
def filtrar_fecha_encoder(df,
                          encoder,
                          fecha=None,
                          hora=None):
    if fecha==None:
        df = df[(df["Hora"] == hora) & (df["Encoder"] == int(encoder))]
    else:
        if fecha == "15-01-2022":
            date = "2022-01-15"
        elif fecha == "31-01-2022":
            date = "2022-01-31"
        elif fecha == "15-02-2022":
            date = "2022-02-15"
        elif fecha == "28-02-2022":
            date = "2022-02-28"
        elif fecha == "15-03-2022":
            date = "2022-03-15"
        elif fecha == "31-03-2022":
            date = "2022-03-31"
        elif fecha == "15-04-2022":
            date = "2022-04-15"
        elif fecha == "30-04-2022":
            date = "2022-04-30"
        elif fecha == "15-05-2022":
            date = "2022-05-15"
        elif fecha == "31-05-2022":
            date = "2022-05-31"
        else:
            date = "2022-06-15"

        df = df[(df["datetime"] == date) & (df["Encoder"] == int(encoder))]

    return df



def plot_prediction_plotly_diff(data,
                                hora_seleccionada,
                                df_predicciones,
                                df_test_metrics,
                                df_attention,
                                prediction_length):


    actuals = data[data["Hora"] == hora_seleccionada].reset_index().iloc[:]["valueDiary"].iloc[-df_attention.shape[0]-2:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        specs=[[{"secondary_y": True}],
                               [{"secondary_y": False}]],
                        vertical_spacing=0.1,
                        subplot_titles=("Predicción para la hora " + str(hora_seleccionada),
                                        ("MAPE " if metrica_elegica else "MAE ") + "para todas las horas "),
                        x_title="Hora",
                        row_width=[0.25, 0.75])

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada].reset_index().iloc[-df_attention.shape[0]-2:]["datetime"],
        y=actuals,
        hovertemplate=
        'Precio real: %{y:.4f}€',
        line_color='blue',
        showlegend=True,
        name='Precio real'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil 98"],
        hovertemplate=
        'Percentil 98: %{y:.4f}€',
        line_color="black",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil 98'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil 90"],
        hovertemplate=
        'Percentil 90: %{y:.4f}€',
        line_color="red",
        mode=("lines" if prediction_length > 1 else None),
        # line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Percentil 90'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil 75"],
        hovertemplate=
        'Percentil 75: %{y:.4f}€',
        line_color="darkorange",
        mode=("lines" if prediction_length > 1 else None),
        # line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Percentil 75'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"] == hora_seleccionada]["Percentil 50"],
        hovertemplate=
        'Predicción: %{y:.4f}€',
        line_color="mediumspringgreen",
        showlegend=True,
        name='Predicción'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil 25"],
        hovertemplate=
        'Percentil 25: %{y:.4f}€',
        line_color="darkorange",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil 25'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil 10"],
        hovertemplate=
        'Percentil 10: %{y:.4f}€',
        #fill='tonexty',
        line_color="red",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil 10'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil 2"],
        hovertemplate=
        'Percentil 2: %{y:.4f}€',
        #fill='tonexty',
        line_color="black",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil 2'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(
        go.Scatter(x=data[data["Hora"] == hora_seleccionada].reset_index().iloc[-df_attention.shape[0]-1:-1]["datetime"],
                   y=df_attention["Hora_"+str(hora_seleccionada)],
                   line_color="grey",
                   hovertemplate=
                   'Atención: %{y:.4f}',
                   name="Atención"),

        secondary_y=True,
        row=1, col=1
    )

    fig.update_layout(title=dict(x=0.5),
                      legend={'traceorder': 'normal'},
                      yaxis_title="Precio (€)")

    if metrica_elegica:
        columna = "mape"
    else:
        columna = "mae"
    for h in df_test_metrics["Hora"].unique():
        fig.add_trace(go.Bar(x=pd.Series(h),
                             y=df_test_metrics[df_test_metrics["Hora"]==h][columna],
                             marker_color=("crimson" if h==hora_seleccionada else "blue"),
                             name=("MAPE" if metrica_elegica else "MAE"),
                             hovertemplate=
                             ('MAPE: %{y:.2f}%' if metrica_elegica else 'MAE: %{y:.2f}'),
                             showlegend=False,
                             xaxis="x"),
                    row=2, col=1)
        
    fig['layout']['yaxis3']['title'] = ("MAPE" if metrica_elegica else "MAE")

    fig.update_layout(width=1000, height=650,
                      margin=dict(t=20))
    fig.update_xaxes(dtick=1, row=2, col=1)
        
    fig.update_yaxes(title_text="Atención", secondary_y=True)

    return fig


@st.cache(suppress_st_warning=True, show_spinner=False)
def plot_horas_metricas(df_predicciones,
                        df_test_metrics,
                        df_metricas_horas,
                        texto_hora_dia):

    if modo == "Predicciones por horas":
        col_filtrada = "datetime"
        eje_x = "Día"
        frase = "todos los días a la hora "
        frase_2 = "la hora "

    elif modo == "Predicciones por días":
        col_filtrada = "Hora"
        eje_x = "Hora"
        frase = "todas las horas del día "
        frase_2 = "el día "

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.12,
                        subplot_titles=("Predicción para " + frase + str(texto_hora_dia),
                                        ("MAPE " if metrica_elegica else "MAE ") + "para " + frase[:15]),
                        x_title=eje_x,
                        row_width=[0.25, 0.75])

    fig2.add_trace(go.Scatter(
        x=df_predicciones[col_filtrada],
        y=df_predicciones["Percentil 50"],
        hovertemplate=
        'Predicciones: %{y:.4f}€',
        line_color="green",
        mode="lines+markers",
        showlegend=True,
        name='Predicciones'),
        row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=df_predicciones[col_filtrada],
        y=df_predicciones["Precio real"],
        hovertemplate=
        'Precio real: %{y:.4f}€',
        line_color="blue",
        mode="lines+markers",
        showlegend=True,
        name='Precio real'),
        row=1, col=1)

    fig2.update_xaxes(tickmode="array", tickvals=df_predicciones[col_filtrada],
                      row=1, col=1, showticklabels=True)

    fig2.update_layout(
        title=dict(x=0.5),
        yaxis_title="Precio (€)",
        xaxis_title=eje_x,
        height=550)

    fig2.add_trace(go.Bar(y=df_test_metrics[("mape" if metrica_elegica else "mae")],
                         name=("MAPE" if metrica_elegica else "MAE"),
                         hovertemplate=
                         ('MAPE: %{y:.2f}%' if metrica_elegica else 'MAE: %{y:.2f}'),
                         xaxis="x",
                         showlegend=False),
                  row=2, col=1)
        
    fig2['layout']['yaxis2']['title'] = ("MAPE" if metrica_elegica else "MAE")

    fig2.update_xaxes(tickmode="array", tickvals=df_predicciones[col_filtrada],
                      row=2, col=1)

    fig2.update_layout(width=1000, height=650,
                       margin=dict(t=20))

    df_metricas_horas = df_metricas_horas[["MAE", "Weighted MAE", "Opposite Weighted MAE", "MAPE", "Métricas"]]
    df_metricas_horas.set_index("Métricas", inplace=True)
    df_metricas_horas = df_metricas_horas.round(4)

    fig_tabla = go.Figure(data=[go.Table(
        header=dict(values=list(df_metricas_horas.columns),
                    fill_color='paleturquoise',
                    align='center',
                    font=dict(size=17)),
        cells=dict(values=[df_metricas_horas["MAE"], df_metricas_horas["Weighted MAE"], df_metricas_horas["Opposite Weighted MAE"], df_metricas_horas["MAPE"]],
                   fill_color='lavender',
                   align='center',
                   font=dict(size=17)))
    ])

    fig_tabla.update_layout(
        title_text="Métricas para " + frase_2 + str(texto_hora_dia),
        title=dict(x=0.5),
        yaxis_title="Precio (€)",
        xaxis_title="Fecha",
        height=550)

    return fig2, fig_tabla



modo = st.sidebar.radio(label="Elija el modo de visualización",
                        options=['Predicciones día a día', 'Predicciones por horas', "Predicciones por días"],
                        help="Podrá visualizar, por un lado, las predicciones de todas las horas para cada día. "
                             "Por otro, podrá visualizar las predicciones de todos los días para cada hora.")

if modo == "Predicciones día a día":

    c1, c2 = st.columns((1, 1))

    with c1:
        fecha_elegida = st.selectbox('Seleccione una de las siguientes fechas:',
                                     ('15-01-2022', '31-01-2022',
                                      '15-02-2022', '28-02-2022',
                                      '15-03-2022', '31-03-2022',
                                      '15-04-2022', '30-04-2022',
                                      '15-05-2022', '31-05-2022',
                                      '15-06-2022'))

    with c2:
        encoder_elegido = st.selectbox('Seleccione la longitud del encoder:',
                                       ("4", "7"))

    datos = datos[datos["datetime"]<=fecha_elegida]

    metrica_elegica = st.checkbox(label="MAPE / MAE", help="Si la casilla está activada, se muestra el MAPE, al contrario, se muestra el MAE", value=True)
    hora_elegida = st.select_slider(label="Seleccione una hora", options=np.arange(0,24))

    predicciones = filtrar_fecha_encoder(df=predicciones,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=hora_elegida)
    test_metrics = filtrar_fecha_encoder(df=test_metrics,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=hora_elegida)
    attention = filtrar_fecha_encoder(df=attention,
                                      encoder=encoder_elegido,
                                      fecha=fecha_elegida,
                                      hora=hora_elegida)

    grafico_pred = plot_prediction_plotly_diff(data=datos,
                                               hora_seleccionada=hora_elegida,
                                               df_predicciones=predicciones,
                                               df_test_metrics=test_metrics,
                                               df_attention=attention,
                                               prediction_length=1)

    st.plotly_chart(grafico_pred, use_container_width=True)

        
elif modo == "Predicciones por horas":
    c1, c2 = st.columns((0.1, 1))
    with c1:
        encoder_elegido = st.radio('Seleccione la longitud del encoder:',
                                  ("4", "7"))
    with c2:
        hora_elegida = st.select_slider(label="Seleccione una hora", options=np.arange(0, 24))

    metrica_elegica = st.checkbox(label="MAPE / MAE",
                                  help="Si la casilla está activada, se muestra el MAPE, al contrario, se muestra el MAE",
                                  value=True)
    fecha_elegida = None
    predicciones = filtrar_fecha_encoder(df=predicciones,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=hora_elegida)

    datos = datos[(datos["datetime"].isin(predicciones["datetime"])) & (datos["Hora"]==int(hora_elegida))]

    metricas_horas = pd.read_excel("Métricas_por_horas.xlsx")
    metricas_horas = filtrar_fecha_encoder(df=metricas_horas,
                                           encoder=encoder_elegido,
                                           fecha=fecha_elegida,
                                           hora=hora_elegida)

    test_metrics = filtrar_fecha_encoder(df=test_metrics,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=hora_elegida)

    grafico, metricas = plot_horas_metricas(df_predicciones=predicciones,
                                            df_test_metrics=test_metrics,
                                            df_metricas_horas=metricas_horas,
                                            texto_hora_dia=hora_elegida)


    st.plotly_chart(grafico, use_container_width=True)
    st.plotly_chart(metricas, use_container_width=True)


else:
    c1, c2 = st.columns((0.2, 1))
    with c1:
        encoder_elegido = st.radio('Seleccione la longitud del encoder:',
                                   ("4", "7"))
    with c2:
        fecha_elegida = st.selectbox('Seleccione una de las siguientes fechas:',
                                     ('15-01-2022', '31-01-2022',
                                      '15-02-2022', '28-02-2022',
                                      '15-03-2022', '31-03-2022',
                                      '15-04-2022', '30-04-2022',
                                      '15-05-2022', '31-05-2022',
                                      '15-06-2022'))

    metrica_elegica = st.checkbox(label="MAPE / MAE",
                                  help="Si la casilla está activada, se muestra el MAPE, al contrario, se muestra el MAE",
                                  value=True)

    metricas_horas = pd.read_excel("Métricas por días.xlsx")
    metricas_horas = filtrar_fecha_encoder(df=metricas_horas,
                                           encoder=encoder_elegido,
                                           fecha=fecha_elegida,
                                           hora=None)

    predicciones = filtrar_fecha_encoder(df=predicciones,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=None)
    test_metrics = filtrar_fecha_encoder(df=test_metrics,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=None)


    grafico, graf_met_horas = plot_horas_metricas(df_predicciones=predicciones,
                                                  df_test_metrics=test_metrics,
                                                  df_metricas_horas=metricas_horas,
                                                  texto_hora_dia=fecha_elegida)


    st.plotly_chart(grafico, use_container_width=True)
    st.plotly_chart(graf_met_horas, use_container_width=True)
