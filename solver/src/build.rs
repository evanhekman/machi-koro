use crate::state::{NUM_CARDS, CARD_COSTS, LANDMARK_COSTS, SUPPLY_MAX};

/// A build option: `None` = skip; `Some(i)` where i < NUM_CARDS = buy card i;
/// `Some(i)` where i >= NUM_CARDS = buy landmark (i - NUM_CARDS).
pub type Opt = Option<u8>;

/// Returns (new_coins, new_cards, new_landmarks) after applying opt.
#[inline]
pub fn apply_build(
    coins: u8,
    cards: &[u8; NUM_CARDS],
    landmarks: u8,
    opt: Opt,
) -> (u8, [u8; NUM_CARDS], u8) {
    match opt {
        None => (coins, *cards, landmarks),
        Some(idx) if (idx as usize) < NUM_CARDS => {
            let i = idx as usize;
            let mut nc = *cards;
            nc[i] += 1;
            (coins - CARD_COSTS[i], nc, landmarks)
        }
        Some(idx) => {
            let i = idx as usize - NUM_CARDS;
            (coins - LANDMARK_COSTS[i], *cards, landmarks | (1 << i))
        }
    }
}

pub fn build_options_slice(coins: u8, cards: &[u8; NUM_CARDS], landmarks: u8) -> ([Opt; 16], usize) {
    let mut opts = [None; 16];
    let mut n = 1usize;
    for i in 0..NUM_CARDS {
        if coins >= CARD_COSTS[i] && cards[i] < SUPPLY_MAX {
            opts[n] = Some(i as u8);
            n += 1;
        }
    }
    for i in 0..4usize {
        if coins >= LANDMARK_COSTS[i] && (landmarks & (1 << i)) == 0 {
            opts[n] = Some((NUM_CARDS + i) as u8);
            n += 1;
        }
    }
    (opts, n)
}
