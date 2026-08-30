# Plantilla LaTeX para TFM

Entorno local para escribir y compilar un TFM o libro con LaTeX, usando `lualatex`, `biber` y `latexmk`.

## Compilar

Opcion recomendada si tienes Perl instalado:

```powershell
latexmk main.tex
```

Opcion disponible con MiKTeX aunque no tengas Perl:

```powershell
lualatex -interaction=nonstopmode -synctex=1 -output-directory=build main.tex
biber --input-directory=build --output-directory=build main
lualatex -interaction=nonstopmode -synctex=1 -output-directory=build main.tex
lualatex -interaction=nonstopmode -synctex=1 -output-directory=build main.tex
```

El PDF se genera en:

```text
build/main.pdf
```

## Limpiar archivos generados

```powershell
latexmk -c main.tex
```

## Estructura

- `main.tex`: documento principal.
- `config/`: paquetes, estilo, comandos y portada.
- `capitulos/`: capitulos del trabajo.
- `anexos/`: anexos.
- `bibliografia/referencias.bib`: bibliografia BibLaTeX/Biber.
- `figuras/`: imagenes y graficos.
- `.latexmkrc`: configuracion de compilacion.
- `.vscode/settings.json`: configuracion de VS Code y LaTeX Workshop.
