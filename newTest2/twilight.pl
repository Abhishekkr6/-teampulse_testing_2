use strict;
use warnings;

my @tones = qw(gossamer velvet prism auric);
my $random_index = int(rand(@tones));
my $tempo = int(rand(40)) + 60;
print "Twilight ribbon $tones[$random_index] sways at $tempo bpm\n";
