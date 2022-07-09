import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title='TFT Predicciones', page_icon="⚡", layout='wide', initial_sidebar_state='auto')

css_file = 'style.css'

def local_css(css_file):
    with open(css_file) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css(css_file)

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
test_metrics = pd.read_excel("Métricas/Test_metrics.xlsx")
predicciones = pd.read_excel("Métricas/Predicciones.xlsx")
attention = pd.read_excel("Métricas/Atención.xlsx")


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
        '<i>Precio real</i>: %{y:.4f}€',
        line_color='blue',
        showlegend=True,
        name='Precio real'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil_98"],
        hovertemplate=
        '<i>Percentil 98</i>: %{y:.4f}€',
        line_color="black",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil: 0.98'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil_90"],
        hovertemplate=
        '<i>Percentil 90</i>: %{y:.4f}€',
        line_color="red",
        mode=("lines" if prediction_length > 1 else None),
        # line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Percentil: 0.9'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil_75"],
        hovertemplate=
        '<i>Percentil 75</i>: %{y:.4f}€',
        line_color="darkorange",
        mode=("lines" if prediction_length > 1 else None),
        # line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Percentil: 0.75'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"] == hora_seleccionada]["Percentil_50"],
        hovertemplate=
        '<i>Predicción</i>: %{y:.4f}€',
        line_color="mediumspringgreen",
        showlegend=True,
        name='Predicción'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil_25"],
        hovertemplate=
        '<i>Percentil 25</i>: %{y:.4f}€',
        line_color="darkorange",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil: 0.25'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil_10"],
        hovertemplate=
        '<i>Percentil 10</i>: %{y:.4f}€',
        #fill='tonexty',
        line_color="red",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil: 0.1'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=data[data["Hora"] == hora_seleccionada]["datetime"][-1:],
        y=df_predicciones[df_predicciones["Hora"]==hora_seleccionada]["Percentil_2"],
        hovertemplate=
        '<i>Percentil 2</i>: %{y:.4f}€',
        #fill='tonexty',
        line_color="black",
        mode=("lines" if prediction_length > 1 else None),
        showlegend=True,
        name='Percentil: 0.02'),
        row=1, col=1,
        secondary_y=False)

    fig.add_trace(
        go.Scatter(x=data[data["Hora"] == hora_seleccionada].reset_index().iloc[-df_attention.shape[0]-1:-1]["datetime"],
                   y=df_attention["Hora_"+str(hora_seleccionada)],
                   line_color="grey",
                   hovertemplate=
                   '<i>Atención</i>: %{y:.4f}',
                   name="Atención"),

        secondary_y=True,
        row=1, col=1
    )

    fig.update_layout(title=dict(x=0.5),
                      legend={'traceorder': 'normal'},
                      yaxis_title="Predcio (€)")

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
                             ('<i> MAPE </i>: %{y:.2f}%' if metrica_elegica else '<i> MAE </i>: %{y:.2f}'),
                             showlegend=False,
                             xaxis="x"),
                    row=2, col=1)

    fig.update_layout(width=1000, height=650,
                      margin=dict(t=20))
    fig.update_xaxes(dtick=1, row=2, col=1)

    return fig



modo = st.sidebar.radio(label="Elija el modo de visualización",
                        options=['Predicciones por días', 'Predicciones por horas'],
                        help="Podrá visualizar, por un lado, las predicciones de todas las horas para cada día. "
                             "Por otro, podrá visualizar las predicciones de todos los días para cada hora.")

if modo == "Predicciones por días":

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

    print(attention)
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

    st.plotly_chart(plot_prediction_plotly_diff(data=datos,
                                                hora_seleccionada=hora_elegida,
                                                df_predicciones=predicciones,
                                                df_test_metrics=test_metrics,
                                                df_attention=attention,
                                                prediction_length=1), use_container_width=True)

else:
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

    metricas_horas = pd.read_excel("Métricas/Métricas_por_horas.xlsx")
    metricas_horas = filtrar_fecha_encoder(df=metricas_horas,
                                           encoder=encoder_elegido,
                                           fecha=fecha_elegida,
                                           hora=hora_elegida)
    test_metrics = filtrar_fecha_encoder(df=test_metrics,
                                         encoder=encoder_elegido,
                                         fecha=fecha_elegida,
                                         hora=hora_elegida)

    def plot_horas_metricas(df_datos,
                            df_predicciones,
                            df_test_metrics,
                            df_metricas_horas,
                            hora_seleccionada):

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=False,
                            vertical_spacing=0.12,
                            subplot_titles=("Predicción para todos los días a la hora " + str(hora_seleccionada),
                                            ("MAPE " if metrica_elegica else "MAE ") + "para todos los días "),
                            x_title="Día",
                            row_width=[0.25, 0.75])

        fig2.add_trace(go.Scatter(
            x=df_datos["datetime"],
            y=df_predicciones["Percentil_50"],
            hovertemplate=
            '<i>Predicciones</i>: %{y:.4f}€',
            line_color="green",
            showlegend=True,
            name='Predicciones'),
            row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=df_datos["datetime"],
            y=df_datos["valueDiary"],
            hovertemplate=
            '<i>Precio</i>: %{y:.4f}€',
            line_color="blue",
            showlegend=True,
            name='Precio'),
            row=1, col=1)

        fig2.update_xaxes(tickmode="array", tickvals=df_datos["datetime"],
                          row=1, col=1)


        fig2.update_layout(
            title=dict(x=0.5),
            yaxis_title="Predcio (€)",
            xaxis_title="Fecha",
            height=550)

        df_metricas_horas.drop(["Encoder", "Hora", "Median APE", "MSE", "RMSE", "Weighted MAPE"], axis=1, inplace=True)
        df_metricas_horas.set_index("Métricas", inplace=True)
        df_metricas_horas = df_metricas_horas.round(4)

        fig2.add_trace(go.Bar(y=df_test_metrics[("mape" if metrica_elegica else "mae")],
                             name=("MAPE" if metrica_elegica else "MAE"),
                             hovertemplate=
                             ('<i> MAPE </i>: %{y:.2f}%' if metrica_elegica else '<i> MAE </i>: %{y:.2f}'),
                             xaxis="x",
                             showlegend=False),
                      row=2, col=1)

        fig2.update_xaxes(tickmode="array", tickvals=df_datos["datetime"],
                          row=2, col=1)

        fig2.update_layout(width=1000, height=650,
                           margin=dict(t=20))

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
            title_text="Métricas para la hora " + str(hora_seleccionada),
            title=dict(x=0.5),
            yaxis_title="Predcio (€)",
            xaxis_title="Fecha",
            height=550)

        return fig2, fig_tabla


    grafico, metricas = plot_horas_metricas(df_datos=datos,
                                            df_predicciones=predicciones,
                                            df_test_metrics=test_metrics,
                                            df_metricas_horas=metricas_horas,
                                            hora_seleccionada=hora_elegida)


    st.plotly_chart(grafico, use_container_width=True)
    st.plotly_chart(metricas, use_container_width=True)