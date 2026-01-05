export interface CartPoleOptions {
    gravity?: number;
    massCart?: number;
    massPole?: number;
    length?: number;
    forceMag?: number;
    tau?: number;
    thetaThreshold?: number;
    xThreshold?: number;
}

export type CartPoleStep = { state: number[]; reward: number; done: boolean };

export class CartPoleEnv {
    gravity: number;
    massCart: number;
    massPole: number;
    totalMass: number;
    poleMassLength: number;
    length: number;
    forceMag: number;
    tau: number;
    thetaThreshold: number;
    xThreshold: number;
    state: number[] = [0, 0, 0, 0];

    constructor({
        gravity = 9.8,
        massCart = 1.0,
        massPole = 0.1,
        length = 0.5,
        forceMag = 10.0,
        tau = 0.02,
        thetaThreshold = 360 * Math.PI / 180,
        xThreshold = 2.4,
    }: CartPoleOptions = {}) {
        this.gravity = gravity;
        this.massCart = massCart;
        this.massPole = massPole;
        this.totalMass = massCart + massPole;
        this.poleMassLength = massPole * length;
        this.length = length;
        this.forceMag = forceMag;
        this.tau = tau;
        this.thetaThreshold = thetaThreshold;
        this.xThreshold = xThreshold;
        this.state = [0, 0, 0, 0];
    }

    reset(): number[] {
        this.state = Array(4).fill(0).map(() => (Math.random() * 0.1 - 0.05));
        return this.state;
    }

    step(action: number): CartPoleStep {
        let [x, xDot, theta, thetaDot] = this.state;
        const force = action === 1 ? this.forceMag : -this.forceMag;
        const costheta = Math.cos(theta);
        const sintheta = Math.sin(theta);

        const temp = (force + this.poleMassLength * thetaDot * thetaDot * sintheta)
            / this.totalMass;
        const thetaAcc = (this.gravity * sintheta - costheta * temp) /
            (this.length * (4.0 / 3.0 - this.massPole * costheta * costheta / this.totalMass));
        const xAcc = temp - this.poleMassLength * thetaAcc * costheta / this.totalMass;

        x += this.tau * xDot;
        xDot += this.tau * xAcc;
        theta += this.tau * thetaDot;
        thetaDot += this.tau * thetaAcc;

        this.state = [x, xDot, theta, thetaDot];

        const done = (
            x < -this.xThreshold ||
            x > this.xThreshold ||
            theta < -this.thetaThreshold ||
            theta > this.thetaThreshold
        );

        const reward = done ? 0 : 1;
        return { state: this.state, reward, done };
    }
}

export class CartPoleRenderer {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    env: CartPoleEnv;
    worldWidth: number;
    scale: number;
    cartWidth = 120;
    cartHeight = 70;
    poleWidth = 24;
    poleLen = 280;

    constructor(canvas: HTMLCanvasElement, env: CartPoleEnv) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.env = env;
        this.worldWidth = env.xThreshold * 2;
        this.scale = canvas.width / this.worldWidth;
    }

    draw(state: number[]) {
        const [x, , theta] = state;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const cartX = x * this.scale + this.canvas.width / 2;
        const cartY = this.canvas.height * 0.8;

        ctx.fillStyle = "#333";
        ctx.fillRect(
            cartX - this.cartWidth / 2,
            cartY - this.cartHeight / 2,
            this.cartWidth,
            this.cartHeight
        );

        ctx.save();
        ctx.translate(cartX, cartY - this.cartHeight / 2);
        ctx.rotate(theta);
        ctx.fillStyle = "#CA9865";
        ctx.fillRect(
            -this.poleWidth / 2,
            -this.poleLen + this.poleWidth / 2,
            this.poleWidth,
            this.poleLen
        );
        ctx.restore();
    }
}

// Optional default export for compatibility:
export default { CartPoleEnv, CartPoleRenderer };