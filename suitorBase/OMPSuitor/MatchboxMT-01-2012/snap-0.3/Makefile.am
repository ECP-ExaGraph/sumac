ACLOCAL_AMFLAGS = -I m4
SUBDIRS = src include lib test 
lib_LTLIBRARIES = libsnap.la

include_HEADERS = include/graph_defs.h include/graph_kernels.h \
	include/graph_metrics.h include/graph_partitioning.h \
	include/graph_gen.h \
	include/utils.h include/sprng.h include/sprng_interface.h \
	include/sprng_f.h

libsnap_la_SOURCES = 

libsnap_la_LIBADD = \
    src/graph_generation/libgraph_gen.la \
    src/graph_kernels/libgraph_kernels.la \
    src/graph_metrics/libgraph_metrics.la \
    src/graph_partitioning/libgraph_partitioning.la \
    src/misc/libsnap_misc.la \
    src/sprng2.0/libsprng.la

