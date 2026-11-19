# Check the editable figure sources before each latexmk run.
# An export error stops the build instead of silently using stale artwork.
use Cwd qw(abs_path);
use File::Basename qw(dirname);
my $paper_dir = dirname(abs_path(__FILE__));
system('python3', "$paper_dir/scripts/export_figures.py") == 0
    or die "draw.io PDF export failed; see the message above.\n";
# draw.io emits PDF 1.7; keep the manuscript output at the same version.
$xdvipdfmx = 'xdvipdfmx -E -V 7 -o %D %O %S';
