AM_CPPFLAGS = -I$(top_srcdir)/include

lib_LTLIBRARIES = libsnap.la
libsnap_la_LDFLAGS = 
libsnap_la_SOURCES = 

libsnap_la_LIBADD = \
	$(top_srcdir)/src/graph_generation/libgraph_gen.la \
	$(top_srcdir)/src/graph_kernels/libgraph_kernels.la \
	$(top_srcdir)/src/graph_metrics/libgraph_metrics.la \
	$(top_srcdir)/src/graph_partitioning/libgraph_partitioning.la \
	$(top_srcdir)/src/misc/libsnap_misc.la \
	$(top_srcdir)/src/sprng2.0/libsprng.la
