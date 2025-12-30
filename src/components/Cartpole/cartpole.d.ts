declare module './cartpole.js' {
    export class CartPoleEnv {
        state: number[];
        reset(): number[];
        step(action: number): { state: number[]; reward: number; done: boolean };
    }

    export class CartPoleRenderer {
        constructor(canvas: HTMLCanvasElement, env: CartPoleEnv);
        draw(state: number[]): void;
    }
}

declare module './cartpole' {
    export class CartPoleEnv {
        state: number[];
        reset(): number[];
        step(action: number): { state: number[]; reward: number; done: boolean };
    }

    export class CartPoleRenderer {
        constructor(canvas: HTMLCanvasElement, env: CartPoleEnv);
        draw(state: number[]): void;
    }
}

declare module './Cartpole.js' {
    export class CartPoleEnv {
        state: number[];
        reset(): number[];
        step(action: number): { state: number[]; reward: number; done: boolean };
    }

    export class CartPoleRenderer {
        constructor(canvas: HTMLCanvasElement, env: CartPoleEnv);
        draw(state: number[]): void;
    }
}

declare module './Cartpole' {
    export class CartPoleEnv {
        state: number[];
        reset(): number[];
        step(action: number): { state: number[]; reward: number; done: boolean };
    }

    export class CartPoleRenderer {
        constructor(canvas: HTMLCanvasElement, env: CartPoleEnv);
        draw(state: number[]): void;
    }
}
