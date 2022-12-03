use numpy::Ix2;

use crate::def_py_enum;
use crate::py::*;
use crate::pycontainer;
use crate::unpack_py_enum;
use crate::unpack_py_enum_expect;

#[derive(Clone)]
pub enum ActionContainer {
    Basic(Actions),
    Invest(InvestActions),
    Sharing(SharingActions),
}
impl HasObjectType for ActionContainer {
    fn object_type(&self) -> ObjectType {
        match self {
            ActionContainer::Basic(_) => ObjectType::Basic,
            ActionContainer::Invest(_) => ObjectType::Invest,
            ActionContainer::Sharing(_) => ObjectType::Sharing,
        }
    }
}


#[derive(Clone)]
#[pyclass(name = "Actions")]
pub struct PyActions(pub ActionContainer);
pycontainer!(PyActions(ActionContainer));

impl PyActions {
    pub fn from_data(data: Array<f64, Ix2>) -> Result<Self, String> {
        if data.shape()[1] == 2 {
            Ok(Self(ActionContainer::Basic(Actions::from_array(data)?)))
        } else if data.shape()[1] == 4 {
            Ok(Self(ActionContainer::Invest(InvestActions::from_array(data)?)))
        } else if data.shape()[1] == 6 {
            Ok(Self(ActionContainer::Sharing(SharingActions::from_array(data)?)))
        } else {
            Err(format!(
                "Invalid shape for action type: {:?}",
                data.shape()
            ))
        }
    }

    pub fn data(&self) -> &Array<f64, Ix2> {
        unpack_py_enum! {
            [ActionContainer](actions) = self.get();
            actions.data()
        }
    }
}

#[pymethods]
impl PyActions {
    #[new]
    #[args(inv_s = "None", inv_p = "None", share_s = "None", share_p = "None")]
    pub fn from_inputs(
        xs: Vec<f64>, xp: Vec<f64>,
        inv_s: Option<Vec<f64>>, inv_p: Option<Vec<f64>>,
        share_s: Option<Vec<f64>>, share_p: Option<Vec<f64>>,
    ) -> Self {
        PyActions(
            if let (Some(inv_s), Some(inv_p)) = (inv_s, inv_p) {
                if let (Some(share_s), Some(share_p)) = (share_s, share_p) {
                    let data = stack![
                        Axis(1),
                        Array::from(xs), Array::from(xp),
                        Array::from(inv_s), Array::from(inv_p),
                        Array::from(share_s), Array::from(share_p)
                    ];
                    ActionContainer::Sharing(
                        SharingActions::from_array(data).unwrap()
                    )
                } else {
                    let data = stack![
                        Axis(1),
                        Array::from(xs), Array::from(xp),
                        Array::from(inv_s), Array::from(inv_p),
                    ];
                    ActionContainer::Invest(
                        InvestActions::from_array(data).unwrap()
                    )
                }
            } else {
                let data = stack![Axis(1), Array::from(xs), Array::from(xp)];
                ActionContainer::Basic(
                    Actions::from_array(data).unwrap()
                )
            }
        )
    }

    #[pyo3(name = "data")]
    #[getter]
    pub fn py_data<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_array(py, self.data())
    }

    #[getter]
    pub fn n(&self) -> usize {
        unpack_py_enum! {
            [ActionContainer](actions) = self.get();
            actions.n()
        }
    }

    #[getter]
    pub fn xs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        unpack_py_enum! {
            [ActionContainer](actions) = self.get();
            PyArray1::from_array(py, &actions.xs())
        }
    }

    #[getter]
    pub fn xp<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        unpack_py_enum! {
            [ActionContainer](actions) = self.get();
            PyArray1::from_array(py, &actions.xp())
        }
    }

    #[getter]
    pub fn py_type(&self) -> String {
        format!("{}", self.get().object_type())
    }

    pub fn __str__(&self) -> String {
        unpack_py_enum! {
            [ActionContainer](actions) = self.get();
            format!("{}", actions)
        }
    }
}


def_py_enum!(StrategyContainer(Strategies));

// this is not quite the same as the base Strategies class
// we contain Vec<A: ActionType> and can convert to Vec<PyActions>
// but not the other way around
// the user is not intended to use this class directly
#[derive(Clone)]
#[pyclass(name = "Strategies")]
pub struct PyStrategies(pub StrategyContainer);
pycontainer!(PyStrategies(StrategyContainer));

macro_rules! build_strat_with_type {
    ( $pyactions_list:expr ; $obj_type:ident ) => {
        {
            let mut actions_list = Vec::with_capacity($pyactions_list.len());
            for pyactions in $pyactions_list {
                let container = pyactions.unpack();
                if container.object_type() != ObjectType::$obj_type {
                    return Err(value_error("All actions must be of the same type"));
                }
                let actions = unpack_py_enum_expect!(container => ActionContainer::$obj_type)?;
                actions_list.push(actions);
            }
            Self(StrategyContainer::$obj_type(Strategies::from_actions(actions_list)))
        }
    };
}

#[pymethods]
impl PyStrategies {

    #[staticmethod]
    pub fn from_actions_list(pyactions_list: Vec<PyActions>) -> PyResult<Self> {
        if pyactions_list.len() == 0 {
            return Err(value_error("Actions list must not be empty"));
        }
        let action_type = pyactions_list[0].get().object_type();
        Ok(if action_type == ObjectType::Basic {
            build_strat_with_type!(pyactions_list; Basic)
        }
        else if action_type == ObjectType::Invest {
            build_strat_with_type!(pyactions_list; Invest)
        }
        else if action_type == ObjectType::Sharing {
            build_strat_with_type!(pyactions_list; Sharing)
        }
        else {
            return Err(value_error("Invalid action type"))
        })
    }

    pub fn to_actions_list(&self) -> Vec<PyActions> {
        match self.clone().unpack() {
            StrategyContainer::Basic(strategies) => strategies
                .into_actions()
                .into_iter()
                .map(|s| PyActions(ActionContainer::Basic(s)))
                .collect(),
            StrategyContainer::Invest(strategies) => strategies
                .into_actions()
                .into_iter()
                .map(|s| PyActions(ActionContainer::Invest(s)))
                .collect(),
            StrategyContainer::Sharing(strategies) => strategies
                .into_actions()
                .into_iter()
                .map(|s| PyActions(ActionContainer::Sharing(s)))
                .collect(),
        }
    }

    pub fn __str__(&self) -> String {
        let s_string = self.to_actions_list().iter().enumerate().map(|(t, a)|
            format!("t = {}, {}", t, a.__str__())
        ).collect::<Vec<_>>().join("\n");
        format!("Strategies:\n{}", s_string)
    }
}
