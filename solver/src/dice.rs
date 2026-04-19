pub struct Outcome {
    pub roll: u8,
    pub is_dbl: bool,
    pub prob: f64,
}

pub static DIST_1: &[Outcome] = &[
    Outcome { roll: 1, is_dbl: false, prob: 1.0 / 6.0 },
    Outcome { roll: 2, is_dbl: false, prob: 1.0 / 6.0 },
    Outcome { roll: 3, is_dbl: false, prob: 1.0 / 6.0 },
    Outcome { roll: 4, is_dbl: false, prob: 1.0 / 6.0 },
    Outcome { roll: 5, is_dbl: false, prob: 1.0 / 6.0 },
    Outcome { roll: 6, is_dbl: false, prob: 1.0 / 6.0 },
];

// 2-dice distribution: unique (sum, is_doubles) pairs with combined probability.
pub static DIST_2: &[Outcome] = &[
    Outcome { roll: 2,  is_dbl: true,  prob: 1.0 / 36.0 },
    Outcome { roll: 3,  is_dbl: false, prob: 2.0 / 36.0 },
    Outcome { roll: 4,  is_dbl: false, prob: 2.0 / 36.0 },
    Outcome { roll: 4,  is_dbl: true,  prob: 1.0 / 36.0 },
    Outcome { roll: 5,  is_dbl: false, prob: 4.0 / 36.0 },
    Outcome { roll: 6,  is_dbl: false, prob: 4.0 / 36.0 },
    Outcome { roll: 6,  is_dbl: true,  prob: 1.0 / 36.0 },
    Outcome { roll: 7,  is_dbl: false, prob: 6.0 / 36.0 },
    Outcome { roll: 8,  is_dbl: false, prob: 4.0 / 36.0 },
    Outcome { roll: 8,  is_dbl: true,  prob: 1.0 / 36.0 },
    Outcome { roll: 9,  is_dbl: false, prob: 4.0 / 36.0 },
    Outcome { roll: 10, is_dbl: false, prob: 2.0 / 36.0 },
    Outcome { roll: 10, is_dbl: true,  prob: 1.0 / 36.0 },
    Outcome { roll: 11, is_dbl: false, prob: 2.0 / 36.0 },
    Outcome { roll: 12, is_dbl: true,  prob: 1.0 / 36.0 },
];
