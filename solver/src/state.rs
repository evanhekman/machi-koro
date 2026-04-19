pub const NUM_CARDS: usize = 10;
pub const MAX_COINS: u8 = 52;

pub mod card {
    pub const WHEAT_FIELD: usize = 0;
    pub const RANCH: usize = 1;
    pub const BAKERY: usize = 2;
    pub const CONVENIENCE_STORE: usize = 3;
    pub const FOREST: usize = 4;
    pub const CHEESE_FACTORY: usize = 5;
    pub const FURNITURE_FACTORY: usize = 6;
    pub const MINE: usize = 7;
    pub const APPLE_ORCHARD: usize = 8;
    pub const FRUIT_VEG_MARKET: usize = 9;
}

pub mod lm {
    pub const TRAIN: u8 = 1 << 0;
    pub const MALL: u8 = 1 << 1;
    pub const PARK: u8 = 1 << 2;
    pub const TOWER: u8 = 1 << 3;
}

pub const WIN_LMS: u8 = 0b1111;

// Indexed by card::*
pub const CARD_COSTS: [u8; NUM_CARDS] = [1, 1, 1, 2, 3, 5, 3, 6, 3, 2];
// Indexed by landmark bit position (0=train, 1=mall, 2=park, 3=tower)
pub const LANDMARK_COSTS: [u8; 4] = [4, 10, 16, 22];
pub const SUPPLY_MAX: u8 = 3;

pub const CARD_KEYS: [&str; NUM_CARDS] = [
    "wheat_field",
    "ranch",
    "bakery",
    "convenience_store",
    "forest",
    "cheese_factory",
    "furniture_factory",
    "mine",
    "apple_orchard",
    "fruit_veg_market",
];

pub const LANDMARK_KEYS: [&str; 4] = [
    "train_station",
    "shopping_mall",
    "amusement_park",
    "radio_tower",
];

#[derive(Clone, Copy)]
pub struct AState {
    pub coins: u8,
    pub cards: [u8; NUM_CARDS],
    pub landmarks: u8,
}

impl AState {
    pub fn initial() -> Self {
        let mut cards = [0u8; NUM_CARDS];
        cards[card::WHEAT_FIELD] = 1;
        cards[card::BAKERY] = 1;
        AState { coins: 3, cards, landmarks: 0 }
    }

    #[inline] pub fn is_won(&self) -> bool { self.landmarks == WIN_LMS }
    #[inline] pub fn has_train(&self) -> bool { self.landmarks & lm::TRAIN != 0 }
    #[inline] pub fn has_park(&self) -> bool { self.landmarks & lm::PARK != 0 }
    #[inline] pub fn has_tower(&self) -> bool { self.landmarks & lm::TOWER != 0 }

    /// Pack (state, depth) into a u64 hash key.
    ///
    /// Layout:
    ///   bits  0– 5 : coins        (0..=52,  6 bits)
    ///   bits  6–25 : card counts  (2 bits × 10, each 0..=3)
    ///   bits 26–29 : landmarks    (4-bit bitmask)
    ///   bits 32–39 : depth        (u8)
    #[inline]
    pub fn pack_key(&self, depth: u8) -> u64 {
        let mut k = self.coins as u64;
        for i in 0..NUM_CARDS {
            k |= (self.cards[i] as u64) << (6 + 2 * i);
        }
        k |= (self.landmarks as u64) << 26;
        k |= (depth as u64) << 32;
        k
    }
}
