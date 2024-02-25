import plotly.express as px

def scatter_plot(data_frame, x_column, y_column, color_column=None, title=None, x_title=None, y_title=None):
    """
    Crea uno scatter plot utilizzando Plotly Express.

    Args:
        data_frame (pandas.DataFrame): Il dataframe contenente i dati.
        x_column (str): Il nome della colonna da utilizzare per l'asse x.
        y_column (str): Il nome della colonna da utilizzare per l'asse y.
        color_column (str, optional): Il nome della colonna da utilizzare per il colore.
        title (str, optional): Il titolo del grafico.
        x_title (str, optional): Il titolo dell'asse x.
        y_title (str, optional): Il titolo dell'asse y.

    Returns:
        plotly.graph_objs._figure.Figure: Il grafico scatter.
    """
    fig = px.scatter(data_frame, x=x_column, y=y_column, color=color_column, title=title)
    
    if x_title:
        fig.update_xaxes(title=x_title)
    if y_title:
        fig.update_yaxes(title=y_title)
        
    return fig

def line_plot(data_frame, x_column, y_column, color_column=None, title=None, x_title=None, y_title=None):
    """
    Crea uno scatter plot utilizzando Plotly Express.

    Args:
        data_frame (pandas.DataFrame): Il dataframe contenente i dati.
        x_column (str): Il nome della colonna da utilizzare per l'asse x.
        y_column (str): Il nome della colonna da utilizzare per l'asse y.
        color_column (str, optional): Il nome della colonna da utilizzare per il colore.
        title (str, optional): Il titolo del grafico.
        x_title (str, optional): Il titolo dell'asse x.
        y_title (str, optional): Il titolo dell'asse y.

    Returns:
        plotly.graph_objs._figure.Figure: Il grafico scatter.
    """
    fig = px.line(data_frame, x=x_column, y=y_column, color=color_column, title=title)
    
    if x_title:
        fig.update_xaxes(title=x_title)
    if y_title:
        fig.update_yaxes(title=y_title)
        
    return fig

def plot(data_frame, x_column, y_column, color_column=None, title=None, x_title=None, y_title=None):
    """
    Crea uno scatter plot utilizzando Plotly Express.

    Args:
        data_frame (pandas.DataFrame): Il dataframe contenente i dati.
        x_column (str): Il nome della colonna da utilizzare per l'asse x.
        y_column (str): Il nome della colonna da utilizzare per l'asse y.
        color_column (str, optional): Il nome della colonna da utilizzare per il colore.
        title (str, optional): Il titolo del grafico.
        x_title (str, optional): Il titolo dell'asse x.
        y_title (str, optional): Il titolo dell'asse y.

    Returns:
        plotly.graph_objs._figure.Figure: Il grafico scatter.
    """
    fig = px.plot(data_frame, x=x_column, y=y_column, color=color_column, title=title)
    
    if x_title:
        fig.update_xaxes(title=x_title)
    if y_title:
        fig.update_yaxes(title=y_title)
        
    return fig