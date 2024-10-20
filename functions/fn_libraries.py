def libraries_versions():
    import pandas as pd
    import matplotlib
    import seaborn as sns
    import numpy as np
    import sklearn
    from platform import python_version

    bibliotecas = {
        "Pandas": pd,
        "Matplotlib": matplotlib,
        "Seaborn": sns,
        "NumPy": np,
        "Scikit-Learn": sklearn,
    }

    print(f"Versão do Python: {python_version()}")
    print()

    print(f"{'Biblioteca':^20} | {'Versão':^10}")
    print(f"{'':-^20} | {'':-^10}")

    for nome, biblioteca in sorted(bibliotecas.items()):
        print(f"{nome:<20} | {biblioteca.__version__:>10}")