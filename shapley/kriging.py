import numpy as np
import openturns as ot


class KrigingIndices(object):
    """
    """
    def __init__(self, model, input_distribution):
        self.model = model
        self.input_distribution = input_distribution
        self.dim = input_distribution.getDimension()
        
    def build_model(self, n_sample_kriging=100, basis_type='linear', kernel='matern'):
        """
        """
        dim = self.dim
        input_sample = self.input_distribution.getSample(n_sample_kriging)
        output_sample = self.model(input_sample)
        
        if basis_type == 'linear':
            basis = ot.LinearBasisFactory(dim).build()
            
        if kernel == 'matern':
            covariance = ot.MaternModel(dim)
            
        kriging_algo = ot.KrigingAlgorithm(input_sample, output_sample, covariance, basis)
        kriging_algo.run()
        self.kriging_result = kriging_algo.getResult()
        
    def compute_indices(self, n_sample=200, n_realization=10, n_bootstrap=50, confidence_level=0.95, with_second_order=False):
        """
        """
        input_design = np.asarray(ot.SobolIndicesAlgorithmImplementation.Generate(self.input_distribution, n_sample, with_second_order))
        kriging_vector = ot.KrigingRandomVector(self.kriging_result, input_design)
        output_designs = np.asarray(kriging_vector.getSample(n_realization)).T
        
        first_indices = np.zeros((self.dim, n_realization, n_bootstrap))
        n_design = input_design.shape[0]
        for i_nz in range(n_realization):
            output_design = output_designs[:, i].reshape(-1, 1)
            sensitivity_normal = ot.SaltelliSensitivityAlgorithm(input_design, output_design, n_sample)
            sensitivity_normal.setBootstrapConfidenceLevel(alpha)
            first_indices[:, i_nz, 0] = np.asarray(sensitivity_normal.getFirstOrderIndices())
            for i_b in range(1, n_bootstrap):
                boot_idx = np.random.choice(range(n_design), n_design)
                input_design_boot = input_design[boot_idx]
                output_design_boot = output_design[boot_idx]
                sensitivity_boot = ot.SaltelliSensitivityAlgorithm(input_design_boot, output_design_boot, n_sample)
                sensitivity_boot.setBootstrapConfidenceLevel(confidence_level)
                first_indices[:, i_nz, i_b] = np.asarray(sensitivity_boot.getFirstOrderIndices())
                
        return first_indices