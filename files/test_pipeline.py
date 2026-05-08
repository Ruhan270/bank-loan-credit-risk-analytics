"""
test_pipeline.py
Unit tests for the bank loan credit risk analytics pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from data_generator import generate_loan_dataset


class TestDataGenerator:
    """Tests for synthetic data generation."""

    def setup_method(self):
        self.df = generate_loan_dataset(500)

    def test_row_count(self):
        assert len(self.df) == 500

    def test_required_columns_exist(self):
        required = ['loan_id', 'credit_score', 'annual_income', 'loan_amount',
                    'is_default', 'loan_status', 'debt_to_income_ratio']
        for col in required:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_no_missing_values(self):
        assert self.df.isnull().sum().sum() == 0, "Dataset contains missing values"

    def test_credit_score_range(self):
        assert self.df['credit_score'].between(300, 850).all(), \
            "Credit scores out of valid range"

    def test_interest_rate_range(self):
        assert self.df['interest_rate'].between(0, 35).all(), \
            "Interest rates out of valid range"

    def test_is_default_binary(self):
        assert set(self.df['is_default'].unique()).issubset({0, 1}), \
            "is_default should be binary (0 or 1)"

    def test_loan_status_values(self):
        valid = {'Fully Paid', 'Late (31-120 days)', 'Charged Off'}
        assert set(self.df['loan_status'].unique()).issubset(valid), \
            "Unexpected loan_status values"

    def test_default_rate_realistic(self):
        rate = self.df['is_default'].mean()
        assert 0.05 < rate < 0.50, \
            f"Default rate {rate:.2%} seems unrealistic"

    def test_positive_loan_amounts(self):
        assert (self.df['loan_amount'] > 0).all(), \
            "Loan amounts must be positive"

    def test_positive_income(self):
        assert (self.df['annual_income'] > 0).all(), \
            "Annual income must be positive"

    def test_dti_ratio_range(self):
        assert self.df['debt_to_income_ratio'].between(0, 1).all(), \
            "DTI ratio should be between 0 and 1"

    def test_unique_loan_ids(self):
        assert self.df['loan_id'].nunique() == len(self.df), \
            "Loan IDs are not unique"

    def test_monthly_payment_positive(self):
        assert (self.df['monthly_payment'] > 0).all(), \
            "Monthly payments must be positive"

    def test_employment_type_values(self):
        valid = {'Salaried', 'Self-Employed', 'Business Owner', 'Retired', 'Part-Time'}
        assert set(self.df['employment_type'].unique()).issubset(valid)

    def test_is_default_consistent_with_status(self):
        """is_default=1 must correspond to non-Fully-Paid statuses."""
        default_mask = self.df['is_default'] == 1
        assert not (self.df.loc[default_mask, 'loan_status'] == 'Fully Paid').any(), \
            "Fully Paid loans should not be marked as default"


class TestRiskMetrics:
    """Tests for computed risk metrics."""

    def setup_method(self):
        self.df = generate_loan_dataset(1000)

    def test_higher_dti_higher_default(self):
        """Higher DTI quartile should show higher default rate."""
        self.df['dti_q'] = pd.qcut(self.df['debt_to_income_ratio'], 4, labels=[1, 2, 3, 4])
        rates = self.df.groupby('dti_q', observed=True)['is_default'].mean()
        assert rates[4] > rates[1], \
            "Highest DTI quartile should have higher default rate than lowest"

    def test_lower_credit_score_higher_default(self):
        """Below-600 credit scores should default more than above-750."""
        low  = self.df[self.df['credit_score'] < 600]['is_default'].mean()
        high = self.df[self.df['credit_score'] > 750]['is_default'].mean()
        assert low > high, \
            f"Low credit score default rate ({low:.2%}) should exceed high ({high:.2%})"

    def test_payment_to_income_ratio_computed(self):
        expected = (self.df['monthly_payment'] * 12) / self.df['annual_income']
        np.testing.assert_array_almost_equal(
            self.df['payment_to_income_ratio'], expected.round(4), decimal=3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
