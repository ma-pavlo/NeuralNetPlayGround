import * as R from 'ramda';

type Shift<A extends readonly any[]> = ((...args: A) => void) extends ((...args: [A[0], ...infer R]) => void) ? R : never;

type GrowExpRev<A extends readonly any[], N extends number, P extends readonly (readonly any[])[]> =
  A['length'] extends N ? A
  : {
    0: GrowExpRev<[...A, ...P[0]], N, P>,
    1: GrowExpRev<A, N, Shift<P>>
  }[[...A, ...P[0]][N] extends undefined ? 0 : 1];

type GrowExp<A extends readonly any[], N extends number, P extends readonly (readonly any[])[]> =
  A['length'] extends N ? A
  : {
    0: GrowExp<readonly [...A, ...A], N, [A, ...P]>,
    1: GrowExpRev<A, N, P>
  }[[...A, ...A][N] extends undefined ? 0 : 1];

type SizedTuple<T, N extends number> = N extends 0 ? readonly [] : N extends 1 ? readonly [T] : GrowExp<readonly [T, T], N, readonly [[T]]>;

type TupleWithoutFirst<T extends readonly any[]> = T extends readonly [infer X, ...infer XS] ? XS : never;
type TupleWithoutLast<T extends readonly any[]> = T extends readonly [...infer XS, infer X] ? XS : never;

type IntKey = number |
  0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|
  '0'|'1'|'2'|'3'|'4'|'5'|'6'|'7'|'8'|'9'|'10'|'11'|'12'|'13'|'14'|'15'|'16'|'17'|'18'|'19';
type Ints = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
type Inc<N extends IntKey> = Ints[N] extends undefined ? number : Ints[N];
type Dec<N extends IntKey> = [-1, 0, ...Ints][N] extends undefined ? number : [-1, 0, ...Ints][N];
type Id<N extends IntKey> = [0, ...Ints][N] extends undefined ? number : [0, ...Ints][N];
type Depth<T> = {
  0: 0,
  1: T extends readonly (infer R)[] ? Inc<Depth<R>> : never
}[T extends readonly any[] ? 1 : 0];

type DeepTuple<Dimensions extends readonly number[], T, N extends number = 0> = {
  0: T,
  1: SizedTuple<DeepTuple<Dimensions, T, Inc<N>>, Dimensions[N]>
}[N extends Dimensions['length'] ? 0 : 1];

export class Tensor<Dimensions extends readonly number[]> {
  // prevent duck typing
  private θtensorDimensions?: Dimensions;

  values: any[];

  constructor(values: DeepTuple<Dimensions, number> | Tensor<Dimensions>) {
    this.values = (values as any).values ? (values as any).values : values;
  }

  add(rhs: number | Tensor<[...{ [K in keyof Dimensions]: 1 | Dimensions[K] }]>): Tensor<Dimensions> {
    return 1 as any;
  }

  hadamard(rhs: number | Tensor<[...{ [K in keyof Dimensions]: 1 | Dimensions[K] }]>): Tensor<Dimensions> {
    return 1 as any;
  }

  multiply(rhs: number): Tensor<Dimensions>;
  multiply<R extends readonly number[]>(
    rhs: Tensor<[Dimensions[Dec<Dimensions['length']>], ...R]>
  ): Tensor<[...TupleWithoutLast<Dimensions>, ...R]>;
  multiply<R extends readonly number[]>(
    rhs: number | Tensor<[Dimensions[Dec<Dimensions['length']>], ...R]>
  ): Tensor<Dimensions> | Tensor<[...TupleWithoutLast<Dimensions>, ...R]> {
    return 1 as any;
  }

  protected is1D(tensor: Tensor<any>): tensor is Tensor<[number]> {
    return typeof tensor.values[0] === 'number';
  }

  protected is2D(tensor: Tensor<any>): tensor is Tensor<[number, number]> {
    return typeof tensor.values[0].values?.[0] === 'number';
  }
}

export class Vector<Rows extends number> extends Tensor<[Rows]> {
  values: number[] = super.values;

  get column() {
    return new Matrix<1, Rows>([this.values] as any);
  }

  get row() {
    return new Matrix<Rows, 1>(this.values.map(v => [v]) as any);
  }

  add(rhs: number | Tensor<[1 | Rows]>): Vector<Rows> {
    return new Vector<Rows>(super.add(rhs));
  }

  hadamard(rhs: number | Tensor<[1 | Rows]>): Vector<Rows> {
    return new Vector<Rows>(super.hadamard(rhs));
  }

  multiply(rhs: number): Vector<Rows>;
  multiply(rhs: Tensor<[Rows]>): number;
  multiply<R extends number[]>(rhs: Tensor<[Rows, ...R]>): Tensor<R>;
  multiply<R extends number[]>(rhs: number | Tensor<[Rows]> | Tensor<[Rows, ...R]>): Vector<Rows> | number | Tensor<R> {
    return typeof rhs === 'number' ? new Vector<Rows>(super.multiply(rhs))
      : this.is1D(rhs) ? R.sum(R.zipWith((l, r) => l * r, this.values, rhs.values))
      : super.multiply(rhs);
  }
}

export class Matrix<Rows extends number, Cols extends number> extends Tensor<[Rows, Cols]> {
  values: number[][] = super.values;

  get T() {
    return new Matrix<Cols, Rows>(R.range(0, this.values[0].length).map(j => this.values.map(row => row[j])) as any);
  }

  get rows() {
    return this.values.map(v => new Vector<Cols>(v as any));
  }

  get columns() {
    return this.T.rows;
  }

  add(rhs: number | Tensor<[1 | Rows, 1 | Cols]>): Matrix<Rows, Cols> {
    return new Matrix<Rows, Cols>(super.add(rhs));
  }

  hadamard(rhs: number | Tensor<[1 | Rows, 1 | Cols]>): Matrix<Rows, Cols> {
    return new Matrix<Rows, Cols>(super.hadamard(rhs));
  }

  multiply(rhs: number): Matrix<Rows, Cols>;
  multiply(rhs: Vector<Cols>): Vector<Rows>;
  multiply<R extends number>(rhs: Tensor<[Cols, R]>): Matrix<Rows, R>;
  multiply<R extends number[]>(rhs: Tensor<[Cols, ...R]>): Tensor<[Rows, ...R]>;
  multiply(
    rhs: number | Tensor<[Cols]> | Tensor<[Cols, number]> | Tensor<[Cols, ...number[]]>
  ): Tensor<[Rows]> | Tensor<[Rows, number]> | Tensor<[Rows, ...number[]]> {
    return typeof rhs === 'number' ? new Matrix<Rows, Cols>(super.multiply(rhs))
      : this.is1D(rhs) ? new Vector<Rows>(this.rows.map(row => R.sum(row.hadamard(rhs).values)) as any)
      : this.is2D(rhs) ? new Matrix<Rows, number>(super.multiply(rhs))
      : super.multiply(rhs);
  }
}

export function populate<T, Dimensions extends readonly number[]>(
  getValue: () => T,
  ...[x, ...xs]: Dimensions
): DeepTuple<Dimensions, T> {
  return [...Array(x)].map(_ => xs.length > 0 ? populate(getValue, ...xs) : getValue()) as any;
}

export function zeros<Dimensions extends readonly number[]>(...dimensions: Dimensions) {
  return populate(() => 0, ...dimensions);
}

export function ones<Dimensions extends readonly number[]>(...dimensions: Dimensions) {
  return populate(() => 1, ...dimensions);
}

export function randoms<Dimensions extends readonly number[]>(...dimensions: Dimensions) {
  return populate(() => Math.random(), ...dimensions);
}


// -------------- tests

type InputSize<LayerDimensions extends readonly number[]> = LayerDimensions[0];
type OutputSize<LayerDimensions extends readonly number[]> = LayerDimensions[Dec<LayerDimensions['length']>];

type ActivationFunctions<LayerDimensions extends readonly number[]> = [...{
  [K in keyof LayerDimensions]:
    K extends '0'|0 ? any
    : <R extends readonly number[]>(z: Tensor<R>) => Tensor<R>
}];

type Weights<LayerDimensions extends readonly number[]> = [...{
  [K in keyof LayerDimensions]:
    K extends '0'|0 ? any
    : K extends IntKey ? Matrix<LayerDimensions[Id<K>], LayerDimensions[Dec<K>]> : never
}];

type Biases<LayerDimensions extends readonly number[]> = [...{
  [K in keyof LayerDimensions]:
    K extends '0'|0 ? any
    : LayerDimensions[K] extends number ? Vector<LayerDimensions[K]>
    : never
}];
// type Biases<LayerDimensions extends readonly number[]> = LayerVectors<LayerDimensions> & { 0: any };

type LayerVectors<LayerDimensions extends readonly number[]> = [...{
  [K in keyof LayerDimensions]: LayerDimensions[K] extends number ? Vector<LayerDimensions[K]> : never
}];

class BabyNet<LayerDimensions extends readonly number[]> {
  private weights: Weights<LayerDimensions>;
  private biases: Biases<LayerDimensions>;

  constructor(
    private layerDimensions: LayerDimensions,
    private activations: ActivationFunctions<LayerDimensions>,
    private activationDervivatives: ActivationFunctions<LayerDimensions>,
    private cost: <R extends number[]>(
      a: Tensor<[OutputSize<LayerDimensions>, ...R]>,
      y: Tensor<[OutputSize<LayerDimensions>, ...R]>
    ) => Tensor<[1, ...R]>,
    private costDerivative: <R extends number[]>(
      a: Tensor<[OutputSize<LayerDimensions>, ...R]>,
      y: Tensor<[OutputSize<LayerDimensions>, ...R]>
    ) => Tensor<[OutputSize<LayerDimensions>, ...R]>,
  ) {
    const [firstLayer, ...otherLayers] = layerDimensions;

    this.weights = [
      null,
      ...R.zipWith(
        (prev: number, current: number) => new Matrix<number, number>(zeros(current, prev)),
        R.drop(1, otherLayers) as any,
        R.dropLast(1, otherLayers) as any
      )
    ] as any;

    this.biases = [
      null,
      ...otherLayers.map(current => new Vector<number>(zeros(current)))
    ] as any;
  }

  forward(x: Vector<InputSize<LayerDimensions>>): { as: LayerVectors<LayerDimensions>, zs: LayerVectors<LayerDimensions> } {
    const as = [x];
    const zs = [x];
    for (let l = 1; l < this.layerDimensions.length; l++) {
      const z = this.weights[l].multiply(as[l-1]).add(this.biases[l]);
      zs.push(z);
      as.push(new Vector<any>(this.activations[l](z)));
    }

    return { as, zs } as any;
  }

  backward(
    as: LayerVectors<LayerDimensions>,
    zs: LayerVectors<LayerDimensions>,
    y: Vector<OutputSize<LayerDimensions>>
  ): { ds: LayerVectors<LayerDimensions> } {
    const L = this.layerDimensions.length;
    const ds = {
      [L-1]: this.costDerivative(as[L-1], y).multiply(this.activationDervivatives[L-1](zs[L-1]))
    };

    for (let l = L-2; l > 0; l--) {
      ds[l] = this.weights[l+1].T.multiply(ds[l+1]).hadamard(this.activationDervivatives[l](zs[l]))
    }

    return { ds } as any;
  }

  trainBatch(
    batch: { x: Vector<InputSize<LayerDimensions>>, y: Vector<OutputSize<LayerDimensions>> }[],
    learningRate = 0.0001
  ): void {
    const batchSize = batch.length;

    batch.forEach(({ x, y }) => {
      const { as, zs } = this.forward(x);
      const { ds } = this.backward(as, zs, y);

      for (let l = 1; l < this.layerDimensions.length; l++) {
        const dw = as[l-1].row.multiply(ds[l].column);
        const db = ds[l];

        this.weights[l].add(dw.multiply(-learningRate));
        this.biases[l].add(db.multiply(-learningRate));
      }
    });
  }
}


const baby = new BabyNet(
  [12288, 8, 1] as const,
  [null, (z: Tensor<any>) => z, (z: Tensor<any>) => z],
  [null, (z: Tensor<any>) => z, (z: Tensor<any>) => z],
  (a, y) => a,
  (a, y) => a
);

baby.backward()
