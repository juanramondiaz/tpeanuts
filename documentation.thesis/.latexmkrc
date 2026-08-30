$pdf_mode = 1;
$pdflatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;
$biber = 'biber %O %B';
$out_dir = 'build';
$aux_dir = 'build';
$emulate_aux = 1;
$cleanup_includes_generated = 1;

@generated_exts = (@generated_exts, 'run.xml', 'bcf', 'synctex.gz');
