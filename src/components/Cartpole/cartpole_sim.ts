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
    dpr: number;

    cartWidth = 52;
    cartHeight = 32;
    poleWidth = 11;
    poleLen = 100;

    constructor(canvas: HTMLCanvasElement, env: CartPoleEnv) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.env = env;
        this.worldWidth = env.xThreshold * 2;

        // Backing store (canvas.width/height) may be larger than CSS size.
        const cssW = canvas.clientWidth || canvas.getBoundingClientRect().width || canvas.width;
        this.dpr = cssW ? (canvas.width / cssW) : 1;

        // Compute world-to-screen scaling in CSS pixels (logical units).
        const logicalW = canvas.width / this.dpr;
        this.scale = logicalW / this.worldWidth;
    }

    draw(state: number[]) {
        const [x, , theta] = state;
        const ctx = this.ctx;

        // Draw in logical (CSS) pixels even if backing store is higher-res.
        const logicalW = this.canvas.width / this.dpr;
        const logicalH = this.canvas.height / this.dpr;

        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        ctx.clearRect(0, 0, logicalW, logicalH);

        const cartX = x * this.scale + logicalW / 2;
        const cartY = logicalH * 0.8;


        // Track / ground line (the cart rides on this)
        const trackY = cartY + this.cartHeight / 2;
        ctx.save();
        ctx.strokeStyle = "rgba(0, 0, 0, 0.18)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(24, trackY);
        ctx.lineTo(this.canvas.width - 24, trackY);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "#4e4e4eff";
        ctx.fillRect(
            cartX - this.cartWidth / 2,
            cartY - this.cartHeight / 2,
            this.cartWidth,
            this.cartHeight
        );

        ctx.save();
        ctx.translate(cartX, cartY - this.cartHeight / 2);
        ctx.rotate(theta);
        ctx.fillStyle = "#dad2a7ff";
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