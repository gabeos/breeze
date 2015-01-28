package breeze.linalg

import breeze.generic.UFunc

/**
 * breeze
 * 7/15/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
object dim extends UFunc {
  implicit def implVDim[V <:Tensor[_,_]]: Impl[V,Int] = new Impl[V,Int] {
    override def apply(v: V): Int = v.size
  }

  implicit def implMDim[T, M <: Matrix[T]]: Impl[M,(Int,Int)] = new Impl[M,(Int,Int)] {
    override def apply(v: M): (Int, Int) = (v.rows,v.cols)
  }
}
