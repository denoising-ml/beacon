import pyfolio as pf
from matplotlib.backends.backend_pdf import PdfPages


class PyfolioReport():
    def __init__(self):
        pass

    def createPyfolioReport(self, pyfoliozer, outfilename):
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns_fig = pf.create_returns_tear_sheet(returns, return_fig=True)

        position_fig = pf.create_position_tear_sheet(returns,
                                                     positions,
                                                     estimate_intraday=False,
                                                     return_fig=True)

        txn_figure = pf.create_txn_tear_sheet(returns, positions, transactions,
                                              unadjusted_returns=None, estimate_intraday='infer',
                                              return_fig=True)

        interesting_fig = pf.create_interesting_times_tear_sheet(
            returns, benchmark_rets=None, legend_loc='best', return_fig=True)

        # risk_fig = pf.create_risk_tear_sheet(positions,
        #                                      style_factor_panel=None,
        #                                      sectors=None,
        #                                      caps=None,
        #                                      shares_held=None,
        #                                      volumes=None,
        #                                      percentile=None,
        #                                      returns=returns,
        #                                      transactions=transactions,
        #                                      estimate_intraday='infer',
        #                                      return_fig=True)

        # Generate PDF
        pp = PdfPages(outfilename)
        pp.savefig(position_fig)
        pp.savefig(returns_fig)
        pp.savefig(txn_figure)
        pp.savefig(interesting_fig)
        # pp.savefig(risk_fig)
        pp.close()