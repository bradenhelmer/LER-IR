import java.util.LinkedList;

public class DirectiveListener extends GloryBaseListener {

    private int loopCount = 0;
    private LinkedList<Integer> loopTypeList;

    public DirectiveListener () {
        loopTypeList = new LinkedList<>();
    }


    @Override
    public void enterStatement(GloryParser.StatementContext ctx) {
        //System.out.println("Statement starts:\n");
    }

    @Override
    public void enterL(GloryParser.LContext ctx) {
        super.enterL(ctx);

        if (ctx.loopType == null) {
            return;
        }
        loopTypeList.add(ctx.loopType.getType());
        loopCount++;

        int intend = loopCount;
        while (intend > 0) {
            intend--;
            for (int i = 0; i < intend; i++) {
                System.out.printf(" ");
            }
        }

        if (ctx.loopType.getType() == GloryParser.OTHER) {
            System.out.print("while ");
        } else {
            System.out.print("for ");
        }
    }

    @Override
    public void enterForParam(GloryParser.ForParamContext ctx) {
        super.enterForParam(ctx);

        String firstBlk = "(" + ctx.id().getText() + "=" + ctx.lBound().getText() + ";";
        System.out.print(firstBlk);
        String secondBlk = ctx.id().getText() + "<" + ctx.uBound().getText() + ";";
        System.out.printf(secondBlk);
        String thirdBlk = ctx.id().getText() + "++" + ") {";
        System.out.println(thirdBlk);

    }

    @Override
    public void enterConditionExpression(GloryParser.ConditionExpressionContext ctx) {
        super.enterConditionExpression(ctx);

        System.out.println("(" + idFilter(ctx.getText()) + ") {");
    }

    @Override
    public void exitStatement(GloryParser.StatementContext ctx) {
        super.exitStatement(ctx);

        int intend = loopCount;
        while (intend>0) {
            for (int i = 0; i < intend; i++) {
                System.out.printf(" ");
            }
            intend--;
        }

        int loopType = loopTypeList.getLast();

        String r = idFilter(ctx.r().getText());
        String e = idFilter(ctx.e().getText());
        e = checkSin(e);

        if ((loopType == GloryParser.NORMAL) || (loopType == GloryParser.OTHER)) {
            System.out.println(r + " = " + e);
        }

        if (loopType == GloryParser.SUMMATION) {
            System.out.println(r + " = " + r + " + " + e);
        }

        if (loopType == GloryParser.PRODUCT) {
            System.out.println(r + " = " + r + " * " + e);
        }

        while (loopCount > 0) {
            loopCount--;
            for (int i = 0; i < loopCount; i++) {
                System.out.printf(" ");
            }
            System.out.println("}");
        }
    }

    private static String checkSin(String string) {
        boolean isFound = string.contains("synthetic");

        if (isFound) {
            String refinedString = "sin(";
            int indexCount = 0;
            String[] arr = string.split("_");
            for (char ch : arr[1].toCharArray()) {
                if (ch == '[' ) {
                    break;
                }
                indexCount++;
            }

            String[] idArray = arr[1].substring(0, indexCount).split("");

            String[] indexArray = arr[1].substring(indexCount+1, arr[1].length()-1).split(",");

            for (int i = 0; i < indexArray.length; i++) {
                refinedString = refinedString + idArray[i] + "[" + indexArray[i] + "]+";
            }

            refinedString = refinedString.substring(0, refinedString.length()-1);
            string = refinedString + ")";
        }
        return string;
    }

    private String idFilter(String string) {

        StringBuilder sb = new StringBuilder();

        boolean andSwitch = true; //
        for (char ch : string.toCharArray()) {
            if (ch == '$') { // when encounter a '&', flip
                andSwitch = !andSwitch;
                continue;
            }

            if (andSwitch) {
                sb.append(ch);
            }
        }
        return sb.toString();
    }
}

