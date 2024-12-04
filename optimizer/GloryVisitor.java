package optimizer;

// Generated from Glory.g4 by ANTLR 4.13.1
import org.antlr.v4.runtime.tree.ParseTreeVisitor;

/**
 * This interface defines a complete generic visitor for a parse tree produced
 * by {@link GloryParser}.
 *
 * @param <T> The return type of the visit operation. Use {@link Void} for
 * operations with no return type.
 */
public interface GloryVisitor<T> extends ParseTreeVisitor<T> {
	/**
	 * Visit a parse tree produced by {@link GloryParser#statement}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitStatement(GloryParser.StatementContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#l}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitL(GloryParser.LContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#forParam}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitForParam(GloryParser.ForParamContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#lBound}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitLBound(GloryParser.LBoundContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#uBound}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitUBound(GloryParser.UBoundContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#e}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitE(GloryParser.EContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#conditionExpression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitConditionExpression(GloryParser.ConditionExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#conditionOp}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitConditionOp(GloryParser.ConditionOpContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#condition}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitCondition(GloryParser.ConditionContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#comparison}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitComparison(GloryParser.ComparisonContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#subscript}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitSubscript(GloryParser.SubscriptContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#expression}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitExpression(GloryParser.ExpressionContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#expressionPrime}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitExpressionPrime(GloryParser.ExpressionPrimeContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#addOp}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitAddOp(GloryParser.AddOpContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#term}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitTerm(GloryParser.TermContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#mulOp}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitMulOp(GloryParser.MulOpContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#factor}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitFactor(GloryParser.FactorContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#exprList}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitExprList(GloryParser.ExprListContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#nonEmptyExprList}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitNonEmptyExprList(GloryParser.NonEmptyExprListContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#r}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitR(GloryParser.RContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#id}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitId(GloryParser.IdContext ctx);
	/**
	 * Visit a parse tree produced by {@link GloryParser#number}.
	 * @param ctx the parse tree
	 * @return the visitor result
	 */
	T visitNumber(GloryParser.NumberContext ctx);
}