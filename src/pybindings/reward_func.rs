use crate::{py::*, pycontainer};


#[derive(Clone)]
#[pyclass(name = "RewardFunc")]
pub struct PyRewardFunc {
    pub reward_func: Box<dyn RewardFunc>,
    pub class: &'static str,
}

pycontainer!(PyRewardFunc(reward_func: Box<dyn RewardFunc>));

#[pymethods]
impl PyRewardFunc {
    #[new]
    pub fn default(n: usize) -> Self {
        Self {
            reward_func: Box::new(LinearReward::default(n)),
            class: "LinearReward",
        }
    }

    #[staticmethod]
    pub fn linear_reward(
        win_a: Vec<f64>, win_b: Vec<f64>,
        lose_a: Vec<f64>, lose_b: Vec<f64>
    ) -> Self {
        Self {
            reward_func: Box::new(LinearReward::new(
                Array::from(win_a),
                Array::from(win_b),
                Array::from(lose_a),
                Array::from(lose_b),
            ).unwrap()),
            class: "LinearReward",
        }
    }
}